from flowbot import FlowBot, TTDuration

if __name__ == '__main__':
    PUZZLE_LOG_DIR = 'puzzle_logs'
    VERBOSE = False
    SHOW_IMGS = False
    SHOW_TS = True

    bot = FlowBot(verbose=VERBOSE, log_puzzles=PUZZLE_LOG_DIR)
    bot.solve_single(verbose=VERBOSE, show_imgs=SHOW_IMGS, show_ts=SHOW_TS)
    bot.solve_series(verbose=VERBOSE, show_imgs=SHOW_IMGS, show_ts=SHOW_TS)
    bot.solve_time_trial(duration=TTDuration._30SEC, verbose=VERBOSE, show_ts=SHOW_TS)