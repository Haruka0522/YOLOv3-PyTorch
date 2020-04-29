import tensorflow as tf


class Logger(object):
    def __init__(self, log_dir):
        """logを記録するディレクトリを作成する"""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scale_summary(self, tag, value, step):
        """
        tag,value,stepをログに記録する。
        3つの変数はいずれもスカラー
        """
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step, description=None)
            self.writer.flush()

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """
        tag_value_pairs,stepをログに記録する。
        3つの変数はいずれもスカラー
        """
        with self.writer.as_default():
            for tag, value in tag_value_pairs:
                tf.summary.scalar(tag, value, step=step, description=None)
                self.writer.flush()
