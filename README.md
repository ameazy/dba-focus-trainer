# dba-focus-trainer
Created to script Focus training on Dragonball Asylum. Uses a python image analyzer to detect a predefined set of images (arrows) and input the correlating directions.

# How to Use `dba_focus_trainer.py`

This guide explains how to use the `dba_focus_trainer.py` script to automate arrow key presses in the "Dragonball Asylum" game window.

## Prerequisites

- Python 3.x installed
- Required Python packages:
  - `opencv-python`
  - `numpy`
  - `pyautogui`
  - `mss`
  - `pygetwindow`
- Template images for arrow detection:
  - Place `arrow_left.png`, `arrow_right.png`, `arrow_up.png`, `arrow_down.png` in the `templates/` directory.

## Setup

1. **Install dependencies**  
   Run the following command in your terminal: pip install opencv-python numpy pyautogui mss pygetwindow

2. **Prepare template images**  
Ensure the following files exist in the `templates/` folder:
- `arrow_left.png`
- `arrow_right.png`
- `arrow_up.png`
- `arrow_down.png`

3. **Open the game** 
Start "Dragonball Asylum" and ensure its window is visible and named "Dragonball Asylum".

## Running the Script

1. **Start the script**  
Run: python dba_focus_trainer.py

2. **Select the capture region**  
- The script will focus the game window and display a screenshot.
- Draw a rectangle with your mouse to select the region containing the arrow prompts.
- Press `ENTER` to confirm your selection.
- Press `ESC` to exit.

3. **Automation process**  
- The script will detect arrow prompts in the selected region.
- It will automatically press the corresponding arrow keys.
- If no arrows are detected, it will attempt to restart the focus training in-game by pressing `k` and then `enter`.

4. **Stopping the script** 
- Press `q` in the script window to quit.

## Troubleshooting

- **Game window not found:**  
Ensure the game is running and the window title is "Dragonball Asylum".

- **Missing template images:**  
Make sure all required arrow images are present in the `templates/` directory.

- **Region selection issues:** 
Draw a valid rectangle (width and height > 5 pixels) before pressing `ENTER`.

## Notes

- The script uses image recognition to detect arrow prompts. Template images should closely match the in-game arrows for best results.
- Adjust the `threshold` and `action_delay` variables in the script if detection or timing is unreliable.
