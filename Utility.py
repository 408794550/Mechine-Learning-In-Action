# encoding: utf-8
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(message)s\n'
                           'function_name: %(funcName)s -- '
                           'line: %(lineno)d -- '
                           'file_name:%(filename)s \n'
                           '----------------------------------')
