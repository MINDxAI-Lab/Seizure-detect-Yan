import sys

sys.path.append("../")
from constants import INCLUDED_CHANNELS, FREQUENCY
from data_utils import resampleData, getEDFsignals, getOrderedChannels
from tqdm import tqdm
import argparse
import numpy as np
import os
import pyedflib
import h5py
import scipy
import gc


def resample_all(raw_edf_dir, save_dir):
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory: {save_dir}")
    
    edf_files = []
    for path, subdirs, files in os.walk(raw_edf_dir):
        for name in files:
            if ".edf" in name:
                edf_files.append(os.path.join(path, name))

    failed_files = []
    processed_count = 0
    
    for idx in tqdm(range(len(edf_files))):
        edf_fn = edf_files[idx]

        save_fn = os.path.join(save_dir, edf_fn.split("/")[-1].split(".edf")[0] + ".h5")
        if os.path.exists(save_fn):
            continue
            
        f = None
        try:
            # Open EDF file
            f = pyedflib.EdfReader(edf_fn)

            # Get channel information
            orderedChannels = getOrderedChannels(
                edf_fn, False, f.getSignalLabels(), INCLUDED_CHANNELS
            )
            
            # Read signals
            signals = getEDFsignals(f)
            signal_array = np.array(signals[orderedChannels, :])
            sample_freq = f.getSampleFrequency(0)
            
            # Close EDF file immediately after reading
            f.close()
            f = None
            
            # Resample if necessary
            if sample_freq != FREQUENCY:
                signal_array = resampleData(
                    signal_array,
                    to_freq=FREQUENCY,
                    window_size=int(signal_array.shape[1] / sample_freq),
                )

            # Save to HDF5 file
            with h5py.File(save_fn, "w") as hf:
                hf.create_dataset("resampled_signal", data=signal_array)
                hf.create_dataset("resample_freq", data=FREQUENCY)
                
            # Clean up memory
            del signal_array
            del signals
            
            processed_count += 1
            
            # Force garbage collection every 100 files
            if processed_count % 100 == 0:
                gc.collect()

        except Exception as e:
            # More specific error handling
            print(f"Error processing {edf_fn}: {str(e)}")
            failed_files.append(edf_fn)
            
        finally:
            # Ensure EDF file is closed
            if f is not None:
                try:
                    f.close()
                except:
                    pass

    print("DONE. {} files failed.".format(len(failed_files)))
    if failed_files:
        print("Failed files:")
        for failed_file in failed_files[:10]:  # Show first 10 failed files
            print(f"  {failed_file}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Resample.")
    parser.add_argument(
        "--raw_edf_dir",
        type=str,
        default=None,
        help="Full path to raw edf files.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Full path to dir to save resampled signals.",
    )
    args = parser.parse_args()

    resample_all(args.raw_edf_dir, args.save_dir)
