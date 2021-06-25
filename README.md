These scripts assume that data has been stored in the data/Annotated_withcat/ directories for Position and Velocity respectively.

The data in those directories should be CSV files, one per participant, with headers as specified in prepare_sliding_windows.py
(e.g. a velocity file's column for head tracker quaternion w would be "v Head w", the header for the column specifying Learning success at the user level would be "Learning Success" ).
Data in position should be at 90Hz and will be subsampled to every 15 frames, Data in velocity should have already been subsampled.

Running prepare_sliding_windows.py will create caches of all the window sizes specified therein. Once these files are generated, optimize_hyperparameters will 

Scripts in the [pyLeon](https://github.com/LeonDong1993/pyLeon_public) and [lgp](https://github.com/LeonDong1993/learning-gain-prediction) directories were written by Hailiang Dong in part for [Extracting Velocity-Based User-Tracking Features to Predict Learning Gains in a Virtual Reality Training Application](https://doi.org/10.1109/ISMAR50242.2020.00099)
