# Smart Categorizer
This is the trainable tool that can be used to automate the categorization process of images. It expects that you provide some categorized examples on which it will be trained on and then it do this job on uncategorized data. 


## Usage example

```python --train_dir=/datasetsA/examples/ --target_data=/dir/containing/images --result_dir=/datasets/autoA```

Args:

`--positives` - directory containing positive images to train on
`--negatives` - (optional) directory containing negative images to train on
`--target_data` - path to directory containing uncategorized data
`--save_to` - path to save automatically categorized data
