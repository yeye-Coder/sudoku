import tkinter as tk
from tkinter import StringVar, Frame, filedialog, messagebox
import cv2
import numpy as np
import pytesseract
from PIL import Image
import os

# =======================
# Configuration Section
# =======================

# For Windows users:
# Uncomment and set the correct path to tesseract.exe if necessary
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# For macOS and Linux, Tesseract is usually in the PATH, so no need to set this.

class SudokuGUI:
    def __init__(self, master):
        self.master = master
        master.title("Sudoku Solver")

        # Initialize data structures
        self.cells = {}
        self.variables = {}
        self.fixed_cells = set()  # To keep track of initial puzzle cells
        self.manual_entry_mode = False  # Initialize manual entry mode as OFF
        self.move_history = []  # Stack to keep track of moves for undo functionality

        # Create GUI components
        self.create_widgets()
        self.create_control_buttons()

    def create_widgets(self):
        # Create a main frame to contain the entire grid
        main_frame = Frame(self.master)
        main_frame.grid(row=0, column=0)

        # Create the 3x3 grid of larger blocks
        for block_row in range(3):
            for block_col in range(3):
                big_block = Frame(main_frame, bd=2, relief='solid')
                big_block.grid(row=block_row, column=block_col)
                # Prevent the big block from resizing with its content
                big_block.grid_propagate(False)
                # Set a fixed size for the big block
                big_block.config(width=150, height=150)

                # Create the small 3x3 grid within the big block
                for i in range(3):
                    for j in range(3):
                        row = block_row * 3 + i
                        col = block_col * 3 + j
                        var = StringVar(value='')  # Initialize with empty string
                        label = tk.Label(
                            big_block,
                            textvariable=var,
                            width=2,
                            height=1,
                            borderwidth=1,
                            relief="solid",
                            font=("Arial", 14),
                            bg="white"
                        )
                        label.grid(row=i, column=j, sticky='nsew')
                        label.bind(
                            "<Button-1>",
                            lambda event, row=row, col=col: self.cell_clicked(event, row, col)
                        )
                        self.cells[(row, col)] = label
                        self.variables[(row, col)] = var

                    # Configure row and column weights inside big block
                    big_block.rowconfigure(i, weight=1)
                    big_block.columnconfigure(i, weight=1)

        # Configure the main_frame grid to prevent resizing and set uniform padding
        for i in range(3):
            main_frame.rowconfigure(i, weight=1)
            main_frame.columnconfigure(i, weight=1)

    def create_control_buttons(self):
        # Frame to hold control buttons
        control_frame = Frame(self.master)
        control_frame.grid(row=1, column=0, pady=10)

        # "Solve" button
        solve_button = tk.Button(control_frame, text="Solve", command=self.solve_puzzle)
        solve_button.pack(side='left', padx=5)

        # "Check" button
        check_button = tk.Button(control_frame, text="Check", command=self.check_grid)
        check_button.pack(side='left', padx=5)

        # "Clear" button
        clear_button = tk.Button(control_frame, text="Clear", command=self.clear_grid)
        clear_button.pack(side='left', padx=5)

        # "Back" button
        back_button = tk.Button(control_frame, text="Back", command=self.undo_move)
        back_button.pack(side='left', padx=5)

        # "Load Puzzle" button
        load_button = tk.Button(control_frame, text="Load Puzzle", command=self.load_puzzle)
        load_button.pack(side='left', padx=5)

        # "Manual Entry" button
        self.manual_button = tk.Button(
            control_frame, text="Manual Entry OFF", command=self.toggle_manual_entry
        )
        self.manual_button.pack(side='left', padx=5)

    def cell_clicked(self, event, row, col):
        # If manual entry mode is on or the cell is not fixed, allow editing
        if self.manual_entry_mode or (row, col) not in self.fixed_cells:
            self.show_options(event, row, col)

    def toggle_manual_entry(self):
        # Toggle manual entry mode
        self.manual_entry_mode = not self.manual_entry_mode
        if self.manual_entry_mode:
            self.manual_button.config(text="Manual Entry ON")
        else:
            self.manual_button.config(text="Manual Entry OFF")
        # No need to modify fixed_cells here

    def show_options(self, event, row, col):
        # Create a popup menu with available options
        menu = tk.Menu(self.master, tearoff=0)
        selected = self.variables[(row, col)].get()
        if selected != '':
            # If a number is selected, only '' (empty) and the selected number are options
            options = [selected, '']
        else:
            # If cell is empty, show available options based on Sudoku rules
            options = self.get_available_options(row, col)
        for option in options:
            display_option = option if option != '' else ' '
            menu.add_command(
                label=display_option,
                command=lambda opt=option, row=row, col=col: self.select_option(opt, row, col)
            )
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def select_option(self, option, row, col):
        previous_value = self.variables[(row, col)].get()
        if self.manual_entry_mode:
            # In Manual Entry ON mode
            if option != '':
                # If cell has a value, mark it as fixed
                self.fixed_cells.add((row, col))
            else:
                # If cell is empty, remove from fixed cells
                if (row, col) in self.fixed_cells:
                    self.fixed_cells.remove((row, col))
        else:
            # In Manual Entry OFF mode, do not modify fixed_cells
            # Record the move only if the cell is not fixed
            if (row, col) not in self.fixed_cells:
                self.move_history.append((row, col, previous_value, option))
        # Set the new value
        self.variables[(row, col)].set(option)
        # Update the cell style
        self.update_cell_style(row, col)

    def update_cell_style(self, row, col):
        # Update the cell style based on whether it's a fixed cell or not
        if (row, col) in self.fixed_cells:
            self.cells[(row, col)].config(bg="#e6e6e6")  # Light grey background for fixed cells
        else:
            self.cells[(row, col)].config(bg="white")

    def get_available_options(self, row, col):
        used_values = set()
        # Check row and column
        for i in range(9):
            val_row = self.variables[(row, i)].get()
            val_col = self.variables[(i, col)].get()
            if val_row != '':
                used_values.add(val_row)
            if val_col != '':
                used_values.add(val_col)
        # Check 3x3 block
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                val = self.variables[(r, c)].get()
                if val != '':
                    used_values.add(val)
        # Remaining options are those not used
        return [str(i) for i in range(1, 10) if str(i) not in used_values] + ['']

    def check_grid(self):
        changes_made = True
        while changes_made:
            changes_made = False
            for row in range(9):
                for col in range(9):
                    if self.variables[(row, col)].get() == '':
                        available_options = self.get_available_options(row, col)
                        if len(available_options) == 1:
                            value = available_options[0]
                            previous_value = self.variables[(row, col)].get()
                            self.variables[(row, col)].set(value)
                            # Do not add to fixed_cells
                            # Record the move
                            if (row, col) not in self.fixed_cells:
                                self.move_history.append((row, col, previous_value, value))
                            self.update_cell_style(row, col)
                            changes_made = True

    def undo_move(self):
        if self.move_history:
            row, col, previous_value, current_value = self.move_history.pop()
            self.variables[(row, col)].set(previous_value)
            self.update_cell_style(row, col)
        else:
            messagebox.showinfo("Undo", "No moves to undo.")

    def clear_grid(self, clear_fixed_cells=False):
        # Modified clear function to only clear non-fixed cells when manual entry is OFF
        if self.manual_entry_mode or clear_fixed_cells:
            # If manual entry mode is ON or clearing fixed cells, clear all cells
            for row in range(9):
                for col in range(9):
                    self.variables[(row, col)].set('')
                    self.cells[(row, col)].config(bg="white")
            self.fixed_cells.clear()
            self.move_history.clear()
            # Keep manual entry mode as is
        else:
            # If manual entry mode is OFF, only clear non-fixed cells
            for row in range(9):
                for col in range(9):
                    if (row, col) not in self.fixed_cells:
                        self.variables[(row, col)].set('')
                        self.cells[(row, col)].config(bg="white")
            self.move_history.clear()
            # Manual entry mode remains OFF
            # Do not change self.fixed_cells

    def solve_puzzle(self):
        # Solve the puzzle using backtracking
        if self.manual_entry_mode:
            messagebox.showinfo("Solve", "Please turn off Manual Entry mode before solving.")
            return
        self.move_history.clear()
        success = self.backtrack_solve()
        if not success:
            messagebox.showinfo("Solve", "No solution exists for the current puzzle.")
        else:
            messagebox.showinfo("Solve", "Puzzle solved successfully!")

    def backtrack_solve(self):
        empty = self.find_empty_cell()
        if not empty:
            return True  # Puzzle solved
        row, col = empty
        for num in map(str, range(1, 10)):
            if self.is_valid(num, row, col):
                previous_value = self.variables[(row, col)].get()
                self.variables[(row, col)].set(num)
                self.move_history.append((row, col, previous_value, num))
                self.update_cell_style(row, col)
                self.master.update()  # Update GUI to show progress
                if self.backtrack_solve():
                    return True
                # Backtrack
                self.variables[(row, col)].set('')
                self.move_history.append((row, col, num, ''))
                self.update_cell_style(row, col)
        return False

    def find_empty_cell(self):
        for row in range(9):
            for col in range(9):
                if self.variables[(row, col)].get() == '':
                    return row, col
        return None

    def is_valid(self, num, row, col):
        # Check row
        for i in range(9):
            if self.variables[(row, i)].get() == num:
                return False
        # Check column
        for i in range(9):
            if self.variables[(i, col)].get() == num:
                return False
        # Check 3x3 block
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if self.variables[(r, c)].get() == num:
                    return False
        return True

    def load_puzzle(self):
        # Open file dialog to select an image file
        file_path = filedialog.askopenfilename(
            title="Select Sudoku Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )
        if file_path:
            try:
                grid = self.extract_sudoku_from_image(file_path)
                self.populate_grid(grid)
                messagebox.showinfo("Load Puzzle", "Puzzle loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load puzzle: {e}")

    def extract_sudoku_from_image(self, image_path):
        try:
            # Load the image using OpenCV
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise ValueError("Could not load image. Please check the file path and try again.")

            # Resize the image to increase the resolution
            img = cv2.resize(img, (900, 900))

            # Apply Gaussian blur to reduce noise
            img = cv2.GaussianBlur(img, (5, 5), 0)

            # Adaptive thresholding to create a binary image
            thresh = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )

            # Find contours in the image
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Extract the largest contour assuming it's the Sudoku grid
            sudoku_contour = contours[0]

            # Adjust epsilon for contour approximation
            epsilon = 0.01 * cv2.arcLength(sudoku_contour, True)
            approx = cv2.approxPolyDP(sudoku_contour, epsilon, True)

            # If more than 4 points are detected, approximate to 4 corners
            if len(approx) != 4:
                rect = cv2.boundingRect(approx)
                approx = np.array([
                    [rect[0], rect[1]],
                    [rect[0] + rect[2], rect[1]],
                    [rect[0] + rect[2], rect[1] + rect[3]],
                    [rect[0], rect[1] + rect[3]]
                ], dtype='float32')

            if len(approx) != 4:
                raise ValueError(f"Expected 4 corners but found {len(approx)}.")

            # Ensure approx is reshaped to (4, 2) and is of type float32
            approx = approx.reshape((4, 2)).astype('float32')
            src_pts = approx

            dst_pts = np.array([[0, 0], [450, 0], [450, 450], [0, 450]], dtype='float32')

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img, M, (450, 450))

            # Further processing on the warped image
            warped = cv2.GaussianBlur(warped, (5, 5), 0)
            warped = cv2.adaptiveThreshold(
                warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )

            # Divide the warped image into 9x9 cells
            grid = []
            cell_size = warped.shape[0] // 9
            for i in range(9):
                row = []
                for j in range(9):
                    cell = warped[
                        i * cell_size: (i + 1) * cell_size,
                        j * cell_size: (j + 1) * cell_size
                    ]
                    digit = self.extract_digit(cell, i, j)
                    row.append(digit)
                grid.append(row)
            return grid

        except Exception as e:
            print(f"Error processing image: {e}")
            raise

    def extract_digit(self, cell, row, col):
        try:
            # Crop the central 75% of each cell
            h, w = cell.shape
            margin = int(h * 0.125)
            cell = cell[margin:-margin, margin:-margin]

            # Invert the cell for better OCR (white background, black digits)
            cell = cv2.bitwise_not(cell)

            # Resize the cell to a larger size to improve OCR accuracy
            cell = cv2.resize(cell, (100, 100))

            # Apply Gaussian Blur
            cell = cv2.GaussianBlur(cell, (3, 3), 0)

            # Thresholding to create a binary image
            _, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Apply Morphological Operations
            kernel = np.ones((3, 3), np.uint8)
            cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, kernel)
            cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, kernel)

            # OCR configuration for better digit recognition
            custom_config = r'--oem 3 --psm 10 outputbase digits'

            # Convert the image to PIL format
            pil_image = Image.fromarray(cell)

            # Use pytesseract to recognize the digit
            text = pytesseract.image_to_string(pil_image, config=custom_config)
            text = text.strip()

            # Save the processed cell image for debugging
            digits_dir = 'extracted_digits'
            if not os.path.exists(digits_dir):
                os.makedirs(digits_dir)
            pil_image.save(os.path.join(digits_dir, f'digit_{row}_{col}.png'))

            # Return the recognized digit if valid
            if text.isdigit() and text in '123456789':
                print(f'Cell [{row},{col}] recognized as: {text}')
                return text
            else:
                print(f'Cell [{row},{col}] could not be recognized. Detected Text: "{text}"')
                return ''
        except Exception as e:
            print(f"Error processing cell [{row},{col}]: {e}")
            return ''

    def populate_grid(self, grid):
        self.clear_grid(clear_fixed_cells=True)
        print("Populating grid with recognized digits:")
        for i in range(9):
            for j in range(9):
                value = grid[i][j]
                if value != '':
                    self.variables[(i, j)].set(value)
                    self.fixed_cells.add((i, j))
                    self.update_cell_style(i, j)
                    print(f'Cell [{i},{j}] set to {value}')
                else:
                    print(f'Cell [{i},{j}] is empty.')

if __name__ == "__main__":
    root = tk.Tk()
    gui = SudokuGUI(root)
    root.mainloop()
