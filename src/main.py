#!/usr/bin/env python3

import logging


if __name__ == "__main__":
    logging.basicConfig(level='INFO', format='%(asctime)s [%(name)s/%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    logging.info("cc")
