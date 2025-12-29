import giatools
if __name__ == '__main__':

    tool = giatools.ToolBaseplate(params_required=False)
    tool.add_input_image('input')
    tool.add_output_image('output')
    for sect in tool.run('YX', output_dtype_hint='binary'):
        sect['output'] = (
            sect['input'].data > sect['input'].data.mean()
        )
