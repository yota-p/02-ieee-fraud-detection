from save_log import timer


class Transformer:

    @timer
    def transform(self, df_train, df_test):
        return df_train, df_test
