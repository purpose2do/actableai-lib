def assert_fix_info_list(actual_list, expect_list):
    assert len(actual_list) == len(expect_list)
    for x, y in zip(
        sorted(actual_list, key=lambda x: f"{x.col}_{x.index}"),
        sorted(expect_list, key=lambda x: f"{x.col}_{x.index}"),
    ):
        assert x.col == y.col
        assert x.index == y.index
        assert len(x.options) == len(y.options)
        for x_o, y_o in zip(x.sorted_options, y.sorted_options):
            assert (
                x_o.value == y_o.value
                if type(x_o.value) == str and type(y_o.value) == str
                else round(x_o.value) == round(y_o.value)
            )
            assert x_o.confidence == y_o.confidence
