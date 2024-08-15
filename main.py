from flowbot import FlowBot, TTDuration

if __name__ == '__main__':
    '''Change these settings (refer to README for more information).'''
    PUZZLE_LOG_DIR = False 
    VERBOSE = False
    SHOW_IMGS = False
    SHOW_TS = True
    SOLVE_MODE = 'tt' # 'single', 'series', or 'tt'
    TT_DURATION = TTDuration._30SEC # _30SEC, _1MIN, _2MIN, or _4MIN

    bot = FlowBot(verbose=VERBOSE, log_puzzles=PUZZLE_LOG_DIR)
    if SOLVE_MODE == 'single':
        bot.solve_single(verbose=VERBOSE, show_imgs=SHOW_IMGS, show_ts=SHOW_TS)
    elif SOLVE_MODE == 'series':
        bot.solve_series(verbose=VERBOSE, show_imgs=SHOW_IMGS, show_ts=SHOW_TS)
    elif SOLVE_MODE == 'tt':
        bot.solve_time_trial(duration=TT_DURATION, verbose=VERBOSE, show_ts=SHOW_TS)