# flow-solver
A solver and input-free bot for the video game [Flow Free](https://www.bigduckgames.com/flowfree).

Time Trials demonstration:

[![Time trials demo](https://img.youtube.com/vi/GEJYJOYi5_c/0.jpg)](https://www.youtube.com/watch?v=GEJYJOYi5_c)

Works on macOS only!

How to use:
1. Download the code.
2. `cd` to the root directory of this project.
3. Run `pip install -r requirements.txt`.
4. Configure the run settings in `main.py`.  See below for more information.

Move the mouse to the top left corner of the screen to trigger `pyautogui`'s failsafe, or press `esc` to exit the program while it is running.

Run configurations:
- `PUZZLE_LOG_DIR` (boolean or string): If `False`, the bot will not log puzzles.  If you want to log puzzles, specify a directory instead.  The bot will save puzzle solutions including the path it used to this directory.
- `VERBOSE` (boolean): Whether to print the technical outputs.
- `SHOW_IMGS` (boolean): Whether to show stages of the puzzle processing as images.
- `SHOW_TS` (boolean): Whether to display time taken for each step of each puzzle, as well as the total time for each puzzle.
- `SOLVE_MODE` (string: `'single'`, `'series'`, or `'tt'`): Whether the bot should solve a single puzzle, a series of puzzles, or a time trial, respectively.
    - In `'single'` mode, the bot will assume the screen is a puzzle and solve it, then end the program.
    - In `'series'` mode, the bot will wait until the user presses `space`.  Use this time to navigate to the desired puzzle.  The bot will then start solving as many puzzles as it can in a row.  It will automatically move to the next puzzle.  IMPORTANT: requires internet to be off to avoid ads and click the correct place!
    - In `'tt'` mode, the bot will wait until the user presses `space`.  Use this time to navigate to the desired time trial.  Set the duration using `TT_DURATION`.  The bot will automatically find the puzzle dimensions.  Note that the previous 3 configurations will be limited/disabled in interest of speed.
- `TT_DURATION` (enum: `_30SEC`, `_1MIN`, `_2MIN`, or `_4MIN`): How long the time trial should be.  The bot will attempt to solve until the time ends, or it is stopped prematurely.