import tensorflow as tf


class Logger(object):
    def __init__(self, log_dir):
        """logを記録するディレクトリを作成する"""
        self.writer = tf.summary.FileWriter(log_dir)

    def scale_summary(self, tag, value, step):
        """
        tag,value,stepをログに記録する。
        3つの変数はいずれもスカラー
        """
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """
        tag_value_pairs,stepをログに記録する。
        3つの変数はいずれもスカラー
        """
        summary = tf.Summary(value=[tf.Summary.Value(
            tag=tag, simple_value=value) for tag, value in tag_value_pairs])
        self.writer.add_summary(summary, step)
