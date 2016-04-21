#encoding=utf-8


class EasyTool(object):
    @staticmethod
    def write_file(path, mode, text):
        with open(path, mode) as f:
            f.write(text)
