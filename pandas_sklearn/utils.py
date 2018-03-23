def df_to_dict(df, columns=None):
    """
    Return a python dict from a pandas dataframe, with columns as keys
    :param df: DateFrame
    :return: dict
    """
    if columns:
        assert len(columns) == len(df.columns)
    d = [
        dict([
            (columns[i] if columns else col_name, row[i])
            for i, col_name in enumerate(df.columns)
        ])
        for row in df.values
    ]
    return d


def flaten_json(data):
    """
    Convert a json object to flat dictionary.
    :param data: json
    :return: dict
    """
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = '||'.join(value)
        elif isinstance(value, dict):
            for kk, vv in value.items():
                data['%s_%s' % (key, kk)] = vv
            del data[key]
    return data
