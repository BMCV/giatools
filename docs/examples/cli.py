import giatools
if __name__ == '__main__':

    tool = giatools.ToolBaseplate(params_required=False)
    tool.add_input_image('input')
    tool.add_output_image('output')
    for section in tool.run('ZYX'):
        section['output'] = (
            section['input'].data > section['input'].data.mean()
        )
