from save_log import stop_watch


class Transformer:

    @stop_watch
    def transform(self, df_train, df_test):
        return df_train, df_test
