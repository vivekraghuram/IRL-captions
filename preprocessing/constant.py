class PreProcessingConstant(object):

    """
    This assumes coco original data set is in the following directory:
    - <project-root>
        - data
            - annotations
                - captions_train2014.json
                - captions_val2014.json
            - images
                - train2014
                    - <image_file>.jpg
                    - ...
                - val2014
    """

    data_splits = {'train': 'train2014', 'val': 'val2014'}
    data_directory = 'data'
    annotations_directory = '{}/annotations'.format(data_directory)
    image_directory = '{}/images'.format(data_directory)

    data_output_directory = "datasets/self_process"

    null_token = "<NULL>"
    start_token = "<START>"
    end_token = "<END>"
    unk_token = "<UNK>"
    special_tokens = [null_token, start_token, end_token, unk_token]
    special_token_ids = {null_token: 0, start_token: 1, end_token: 2, unk_token: 3}

