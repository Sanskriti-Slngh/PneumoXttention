# SF2020
A Convolutional Neural Network compensating for Human Fallibility when Detecting Pneumonia through Attention

> ## Step 1
> Download the following file(s):
> - https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data?select=stage_2_train_images
> - https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data?select=stage_2_detailed_class_info.csv
> - https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data?select=stage_2_train_labels.csv

> ## Step 2
> Run **"rsna_data_process"** to generate train and test data in the form of pandaframes with the respective amount of 80% and 20% data.
>
> Change the name of the directories where you downloaded the data (rsna_data_dir) and where you want to save the pd (data_out_dir).

    rsna_data_dir = 'F:/'
    data_out_dir = "../data/"
    
    
> ## Step 3
>  Run **"data_prep.py"** *twice* to generate train, val, and test numpy arrays for sizes 512, 256 and save the data into files for future use. 
>
> Change the name of directories in needed areas and the following variables.

    resolution = 512/256      # run program twice with both these values
    data_in = ("train_data_file", "test_data_file")
    data_out_dir = "dir_you_want_to_save_file_in"

> ## Step 4
> Run **"train.py"** three times to create heatmaps for the train, validation, and test set. 
> ###### If you want to skip this part and access the heatmaps you can send an email to *sanskritisingh0914@gmail.com* requesting the heatmaps.
> 
> Change the following variables to the given values:

    model_name = 'm13'
    generate_heatmap = 'train'/'val'/'test'    # run the program three times with genereate_heatmap and predict_on as 'train', 'val, and 'test'
    predict_on = 'train'/'val'/'test'
    x512 = True
    x256 = False
    use_heatmap = False
    params['use_heatmap'] = False
    params['generate_heatmap'] = True
    params['model_dir'] = 'm13_model_dir'

> ## Step 5
> Run **"train.py"** (original file) to reproduce final results. The output will be the F1 score, AUROC, recall, and precision.
>
> Change the following variables to the given values:

    model_name = 'm13h'
    generate_heatmap = ''
    predict_on = 'test'
    x512 = False
    x256 = True
    use_heatmap = True
    params['use_heatmap'] = True
    params['generate_heatmap'] = False
    params['model_dir'] = 'm13h_model_dir'

## For any questions or concerns with the code or procedure please email *sanskritisingh0914@gmail.com* with these questions
