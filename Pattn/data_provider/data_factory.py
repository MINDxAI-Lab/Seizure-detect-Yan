from data_provider.data_loader import   Dataset_Custom,  Dataset_ETT_hour, Dataset_ETT_minute, Dataset_EEG_Seizure
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
    'ett_h': Dataset_ETT_hour,
    'ett_m': Dataset_ETT_minute,
    'eeg_seizure': Dataset_EEG_Seizure,
}
def data_provider(args, flag, drop_last_test=True, train_all=False, train_scaler=None):
    """
    Factory function to create and configure a dataset and its corresponding DataLoader.

    This function selects the appropriate Dataset class based on `args.data`, configures
    it with parameters from `args`, and then wraps it in a DataLoader. It handles
    different configurations for training, validation, and test sets.
    """
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent
    max_len = getattr(args, 'max_len', -1)
    
    # Configure DataLoader parameters based on the dataset split (train, val, or test)
    if flag == 'test':
        shuffle_flag = False
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'val':
        shuffle_flag = False  # Don't shuffle validation data for stable evaluation
        drop_last = False     # Don't drop last batch to use all validation samples
        batch_size = args.batch_size
        freq = args.freq
    else: # flag == 'train'
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    
    # Handle different parameter sets for different datasets
    if args.data == 'eeg_seizure':
        # For EEG seizure detection dataset, pass all relevant parameters, including for augmentation
        data_set = Data(
            model_id=args.model_id,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, getattr(args, 'label_len', 0), getattr(args, 'pred_len', 0)],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            max_len=max_len,
            train_all=train_all,
            train_ratio=getattr(args, 'train_ratio', 1.0),
            shared_scaler=train_scaler,  # Pass the training scaler for val/test to ensure consistent scaling
            
            # Pass decomposition augmentation parameters
            aug_decomp=getattr(args, 'aug_decomp', False),
            aug_p=getattr(args, 'aug_p', 0.5),
            aug_win=getattr(args, 'aug_win', 129),
            aug_scale_low=getattr(args, 'aug_scale_low', 0.9),
            aug_scale_high=getattr(args, 'aug_scale_high', 1.1),
            aug_noise=getattr(args, 'aug_noise', 0.02),
            aug_only_bg=getattr(args, 'aug_only_bg', False),
            
            # Pass Window Warping augmentation parameters
            aug_ww=getattr(args, 'aug_ww', False),
            aug_ww_p_low=getattr(args, 'aug_ww_p_low', 0.3),
            aug_ww_p_high=getattr(args, 'aug_ww_p_high', 0.7),
            aug_ww_win_ratio_low=getattr(args, 'aug_ww_win_ratio_low', 0.1),
            aug_ww_win_ratio_high=getattr(args, 'aug_ww_win_ratio_high', 0.3),
            aug_ww_speed_low=getattr(args, 'aug_ww_speed_low', 0.8),
            aug_ww_speed_high=getattr(args, 'aug_ww_speed_high', 1.2),
            aug_ww_margin=getattr(args, 'aug_ww_margin', 0.5),
            aug_ww_only_bg=getattr(args, 'aug_ww_only_bg', False)
        )
    else:
        # For other standard time-series forecasting datasets
        data_set = Data(
            model_id=args.model_id , 
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, getattr(args, 'label_len', 0), getattr(args, 'pred_len', 0)],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            max_len=max_len,
            train_all=train_all
        )
        
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
