from pprint import pprint


class Logger:
    def __init__(self) -> None:
        print('Initialised logger class')

    def log_message(self, *msgs):
        print('-'*100)
        for msg in msgs:
            if (isinstance(msg, dict) or isinstance(msg, tuple) or isinstance(msg, list)):
                pprint(msg)
            else:
                print(msg)
