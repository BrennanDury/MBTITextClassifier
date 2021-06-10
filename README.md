# MBTITextClassifier
https://drive.google.com/drive/folders/10JZk8ea3OxWs7CA1jNFj7WsmrzyDD3qr
Data and results are stored here.
In the Data folder is a file named natural_language.csv as created by running collector.py for the comments, types, and dimensions. There is also an equivalent file for testing named test_natural_language.csv. This file has one comment for each personality type. There is also a file 1000natural_language.csv which is the natural_language.csv filtered to only include comments with length at least 1000. This was used for final results.
In the Results folder is Test_Results and Final_Results. The files in Test_Results is the result of running analyzer.py on test_natural_language.csv. The files in Final_Results is the result of running analyzer.py on 1000natural_language.csv. The folders Data_Plots and Test_Data_Plots have visualizations of various statistics about the data and the models. The Models and Test_Models folders contain the models themselves, alongside plots of a tree from each random forest model and a json file of the results of each model. For both types of models, the files list the testing accuracy followed by the training accuracy. The random forest models also list the complete feature importances, but the top 20 features are also plotted more understandably in Data_Plots and Test_Data_Plots. Note that test files relating not just to data statistics but to the models are meaningless because training a model on tiny amounts of data is worthless.
The runtime for both collector.py and analyzer.py may be long.
To replicate collecting the data, download and run collector.py in the environment of your choice. Make sure pmaw, re, datetime, and csv are installed.
To replicate analyzing the data, you need analyzer.py and a target folder for data plots and for models. Make sure you have installed json, itemgetter, numpy, pandas, matplotlib.pyplot, seaborn, sklearn, pickle, and nltk. NLTK may ask you a few times to manually enter some python code for downloads, follow its instructions. You can delete that code after downloading. Specify in the global variables of analyzer.py folder, data_folder, and nl_file_name the target destination of the files. By default, folder equals an empty or non-empty folder named '/Models', data_folder equals an empty or non-empty folder named 'Data_Plots', and nl_file equals 1000natural_language.csv. The two folders must already exist at runtime. Run analyzer.py. If you would like to search for uses of a specific word, change the target_word global variable to that word. If you only want to run a part of the analysis, comment out methods in main.
