import sys
import logging


def configure_logging():
    logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)],
                        format='[%(asctime)s : %(levelname)s : %(message)s]',
                        level=logging.DEBUG)

