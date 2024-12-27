# Devin Zhang

import os
import zipfile
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt

def load_openloris_data(data_path, max_images_per_object=3):
    """
    Load the OpenLORIS-Object dataset from a zip file, with an option to limit the number of images per object.

    Parameters:
    data_path (str): Path to the directory containing the 'test.zip' file of the OpenLORIS-Object dataset.
    max_images_per_object (int, optional): Maximum number of images to load per object. If None, load all images.

    Returns:
    dict: A nested dictionary with the following structure:
        {
            'test': {
                'rgb': {
                    'factor': {
                        'segment': {
                            'object': np.array of shape (n_frames, height, width, 3)
                        }
                    }
                }
            }
        }
        Where:
        - 'factor' is one of: 'illumination', 'occlusion', 'pixel', 'clutter'
        - 'segment' is like 'segment1', 'segment2', etc.
        - 'object' is like 'bottle_01', 'cup_02', etc.
        - The numpy array contains the RGB image data for each frame of the object

    Raises:
    FileNotFoundError: If the 'test.zip' file is not found at the specified path.

    Note:
    - This function assumes a specific structure in the zip file:
      test/factor/segment/object/frame.jpg
    - Frames with inconsistent shapes within an object should be kept as lists instead of numpy arrays.
    - The function should print progress and warning messages during the loading process.
    """
    #Initialize our dict
    dataset = {'test': {'rgb': {}}}

    # Ensure the zip file exists
    zip_file_path = os.path.join(data_path, 'test.zip')
    if not os.path.exists(zip_file_path):
        raise FileNotFoundError(f"'{zip_file_path}' not found at the provided path: {data_path}")

    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zf:
        # List all files in the zip archive
        file_list = zf.namelist()

        # Filter for image files only and process each
        for file_path in file_list:
            if file_path.endswith('.jpg'):
                parts = file_path.split('/')

                # Adjust to handle cases where there are more than 5 components in the path
                if len(parts) == 5:
                    _, factor, segment, object, frame = parts
                else:
                    continue

                # Initialize our nested dictionary structure if the component not already present
                if factor not in dataset['test']['rgb']:
                    dataset['test']['rgb'][factor] = {}
                if segment not in dataset['test']['rgb'][factor]:
                    dataset['test']['rgb'][factor][segment] = {}
                if object not in dataset['test']['rgb'][factor][segment]:
                    dataset['test']['rgb'][factor][segment][object] = []

                # Read the image if we havent reached max images yet
                if max_images_per_object is None or len(dataset['test']['rgb'][factor][segment][object]) < max_images_per_object:
                    with zf.open(file_path) as image_file:
                        try:
                            image = Image.open(io.BytesIO(image_file.read()))
                            image = image.convert('RGB')  # Ensure RGB format
                            dataset['test']['rgb'][factor][segment][object].append(np.array(image))

                        except Exception as e:
                            print(f"Warning: Could not load image {file_path}. Error: {e}")

    
    # Ensure consistent shapes within the object
        for object in dataset['test']['rgb'][factor][segment]:
            frames = dataset['test']['rgb'][factor][segment][object]
            if all(frame.shape == frames[0].shape for frame in frames):
                dataset['test']['rgb'][factor][segment][object] = np.array(frames)  # Convert to numpy array if shapes are consistent
            else:
                dataset['test']['rgb'][factor][segment][object] = frames  # Keep as list if shapes vary


    print("Dataset loading complete.")
    return dataset

def preprocess_for_dmd(dataset, factor, segment, object_name):
    """
    Preprocess data for a specific object under a given factor and segment for DMD analysis.

    Parameters:
    dataset (dict): The dataset loaded by load_openloris_data
    factor (str): The factor to analyze (e.g., 'illumination', 'occlusion', 'pixel', 'clutter')
    segment (str): The segment to analyze (e.g., 'segment1', 'segment2', etc.)
    object_name (str): The object to analyze (e.g., 'bottle_01', 'cup_02', etc.)

    Returns:
    np.array: Preprocessed data suitable for DMD, shape (n_pixels * 3, n_frames)
    where n_pixels is the number of pixels in each frame (height * width),
    3 represents the RGB channels, and n_frames is the number of frames.

    Note:
    - This function should flatten each frame and stack them as columns.
    - The returned array should be of type float64 and normalized to the range [0, 1].
    """
    # Retrieve frames for the specified object
    frames = dataset['test']['rgb'][factor][segment].get(object_name, None)

    if frames is None:
        raise ValueError(f"Object '{object_name}' not found in factor '{factor}', segment '{segment}'.")

    # Ensure frames are either a list of NumPy arrays or a single NumPy array
    if isinstance(frames, list):
        frames = np.array(frames)  # Convert list of arrays to a single NumPy array if consistent

    # Normalize the image data to range [0, 1]
    frames = frames.astype(np.float64) / 255.0

    # Reshape each frame to a flattened vector and stack them as columns
    n_frames = frames.shape[0]  # Number of frames
    height, width, channels = frames.shape[1:]
    n_pixels = height * width

    # Flatten each frame and stack as columns
    preprocessed_data = frames.reshape(n_frames, n_pixels * channels).T

    return preprocessed_data

def analyze_and_plot_environmental_factor(data, factor, original_shape, r=2, plot=True):
    """
    Perform DMD analysis and plot eigenvalues and modes for a specific environmental factor.

    Parameters:
    data (np.array): Preprocessed data, shape (n_features, n_frames), dtype float64
        where n_features = height * width * 3 (for RGB channels)
        and n_frames is the number of frames
    factor (str): Environmental factor being analyzed. One of:
        'illumination': Changes in lighting conditions
        'occlusion': Partial blocking of objects
        'pixel': Changes in object size/resolution
        'clutter': Different background conditions
    original_shape (tuple): Original shape of a single RGB image (height, width, 3)
    r (int): Number of modes to compute and plot. Default is 2.

    Returns:
    tuple: (modes, eigenvalues, dynamics)
        modes (np.array): DMD modes, shape (n_features, r), dtype complex128
        eigenvalues (np.array): DMD eigenvalues, shape (r,), dtype complex128
        dynamics (np.array): DMD mode dynamics, shape (r, n_frames-1), dtype complex128

    Visualization:
    The function creates a figure with two subplots:
    1. DMD Eigenvalues Plot (left):
       - Scatter plot of complex eigenvalues
       - X-axis: Real part
       - Y-axis: Imaginary part
       - Dashed lines at x=1 and y=0 for reference
       - Title includes the environmental factor

    2. First DMD Mode Plot (right):
       - Visualization of the first DMD mode reshaped to original image dimensions
       - Values normalized to [0,1] for better visualization
       - Includes a colorbar showing the scale
       - Title includes the environmental factor
       - No axis labels (image display)

    Note:
    - The function performs both DMD computation and visualization
    - Only the first mode is visualized, though r modes are computed
    - The plots help understand how DMD captures patterns under different environmental conditions
    - Eigenvalues near the unit circle indicate persistent patterns
    - Mode visualization shows spatial patterns captured by DMD
    """
    # Perform Singular Value Decomposition (SVD)
    X1 = data[:, :-1]  # First n-1 frames
    X2 = data[:, 1:]   # Last n-1 frames
    U, Sigma, Vt = np.linalg.svd(X1, full_matrices=False)

    # Truncate to rank r
    U_r = U[:, :r]
    Sigma_r = np.diag(Sigma[:r])
    Vt_r = Vt[:r,:]

    # Compute the DMD matrix
    A_tilde = U_r.T @ X2 @ Vt_r.T @ np.linalg.inv(Sigma_r)

    # Compute eigenvalues and modes
    # Φ = X2 * V * S^(-1) * W
    eigenvalues, W = np.linalg.eig(A_tilde)
    modes = X2 @ Vt_r.T @ np.linalg.inv(Sigma_r) @ W

    # Compute time, mode dynamics for all eigenvalues
    time_powers = np.vstack([np.power(eigenvalue, np.arange(data.shape[1] - 1)) for eigenvalue in eigenvalues])
    dynamics = (np.linalg.inv(W) @ (U_r.conj().T @ data[:, :-1])) * time_powers

    # Visualization
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Eigenvalues plot
        axs[0].scatter(eigenvalues.real, eigenvalues.imag, color='blue', label='Eigenvalues')
        axs[0].add_artist(plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='dashed'))
        axs[0].axhline(0, color='black', linewidth=0.5, linestyle='dashed')
        axs[0].axvline(1, color='black', linewidth=0.5, linestyle='dashed')
        axs[0].set_xlabel('Real Part')
        axs[0].set_ylabel('Imaginary Part')
        axs[0].set_title(f'{factor.capitalize()} - DMD Eigenvalues')
        axs[0].legend()

        # First mode plot
        first_mode = np.abs(modes[:, 0])  # Take the magnitude of the first mode
        first_mode_image = first_mode.reshape((*original_shape[:2], 3))  # Reshape to include RGB dimensions
        first_mode_image_normalized = (first_mode_image - first_mode_image.min()) / (first_mode_image.max() - first_mode_image.min())  # Normalize to [0, 1]
        im = axs[1].imshow(first_mode_image_normalized, cmap='viridis')
        axs[1].set_title(f'{factor.capitalize()} - First DMD Mode')
        axs[1].axis('off')
        fig.colorbar(im, ax=axs[1], orientation='vertical')

        plt.tight_layout()
        plt.show()

    return modes, eigenvalues, dynamics

def plot_environmental_factor_dynamics(dynamics, eigenvalues, factor, r=2):
    """
    Plot the dynamics of DMD modes for a specific environmental factor.

    Parameters:
    dynamics (np.array): DMD mode dynamics, shape (r, n_frames-1), dtype complex128
        where r is the number of modes and n_frames is the number of frames
    eigenvalues (np.array): DMD eigenvalues, shape (r,), dtype complex128
    factor (str): Environmental factor being analyzed. One of:
        'illumination': Changes in lighting conditions
        'occlusion': Partial blocking of objects
        'pixel': Changes in object size/resolution
        'clutter': Different background conditions
    r (int): Number of modes to plot. Default is 2.

    Returns:
    None: The function displays the plots but does not return any value

    Visualization:
    The function creates a figure with two subplots:
    1. Mode Dynamics Plot (left):
       - X-axis: Time steps
       - Y-axis: Mode magnitude (log scale)
       - One line per mode, with different colors
       - Legend identifying each mode
       - Title includes the environmental factor

    2. Eigenvalue Plot (right):
       - Scatter plot of complex eigenvalues
       - X-axis: Real part
       - Y-axis: Imaginary part
       - Unit circle shown for reference
       - Title includes the environmental factor

    Note:
    - The dynamics plot shows how each mode evolves over time
    - The eigenvalue plot helps interpret the temporal behavior:
      * Eigenvalues inside unit circle: decaying modes
      * Eigenvalues outside unit circle: growing modes
      * Eigenvalues near unit circle: persistent modes
    """
    n_frames = dynamics.shape[1] + 1  # Total number of frames (including initial state)
    time = np.arange(n_frames - 1)  # Time steps for dynamics

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Mode Dynamics Plot
    for i in range(r):
        axs[0].plot(time, np.abs(dynamics[i, :]), label=f'Mode {i+1}')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Time Steps')
    axs[0].set_ylabel('Mode Magnitude (log scale)')
    axs[0].set_title(f'{factor.capitalize()} - Mode Dynamics')
    axs[0].legend()

    # Eigenvalue Plot
    axs[1].scatter(eigenvalues.real, eigenvalues.imag, color='blue', label='Eigenvalues')
    unit_circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='dashed')
    axs[1].add_artist(unit_circle)
    axs[1].axhline(0, color='black', linewidth=0.5, linestyle='dashed')
    axs[1].axvline(0, color='black', linewidth=0.5, linestyle='dashed')
    axs[1].set_xlabel('Real Part')
    axs[1].set_ylabel('Imaginary Part')
    axs[1].set_title(f'{factor.capitalize()} - DMD Eigenvalues')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def compare_environmental_factors(dataset, object_name, factors=['illumination', 'occlusion', 'pixel', 'clutter'], segment='segment1', r=2):
    """
    Compare DMD analysis results across different environmental factors for the same object.

    Parameters:
    dataset (dict): The loaded OpenLORIS dataset
    object_name (str): Name of the object to analyze (e.g., 'bottle_01')
    factors (list): List of factors to compare
    segment (str): Segment to analyze. Default is 'segment1'
    r (int): Number of DMD modes to compute. Default is 2.

    Returns:
    dict: A dictionary containing comparison metrics for each environmental factor
        {
            'illumination': {
                'eigenvalue_magnitudes': np.array of shape (r,)
                    Magnitude of DMD eigenvalues, indicating pattern stability
                    - Values near 1 indicate stable patterns
                    - Values < 1 indicate decaying patterns
                    - Values > 1 indicate growing patterns
                
                'mode_energies': np.array of shape (r,)
                    Energy content of each DMD mode
                    - Higher values indicate more dominant patterns
                    - Lower values indicate less significant patterns
                    - Sum of energies indicates total pattern strength
                
                'temporal_variation': np.array of shape (r,)
                    Standard deviation of mode dynamics over time
                    - Higher values indicate more temporal change
                    - Lower values indicate more temporal stability
                    - Helps identify which patterns are most variable
            },
            .. entries for other factors ...
        }

    Calculation Hints:
    1. Eigenvalue magnitudes:
       - For complex eigenvalues λ, magnitude = |λ| = √(real² + imag²)
       - Using numpy: np.abs(eigenvalues)
       Example:
       ```python
       eigenvalue_magnitudes = np.abs(eigenvalues)  # shape: (r,)
       
    2. Mode energies:
       - For complex modes, sum absolute values across spatial dimensions
       - Using numpy: np.sum(np.abs(modes), axis=0)
       Example:
       ```python
       mode_energies = np.sum(np.abs(modes), axis=0)  # shape: (r,)
       

    3. Temporal variation:
       - Calculate standard deviation of dynamics over time
       - Using numpy: np.std(np.abs(dynamics), axis=1)
       Example:
       ```python
       temporal_variation = np.std(np.abs(dynamics), axis=1)  # shape: (r,)
       

    Visualization:
    The function creates a figure with subplots for each factor:
    1. Original Image (left column):
       - Shows the first frame under each condition
    
    2. First DMD Mode (middle column):
       - Shows the spatial pattern of the first mode
       - Normalized to [0,1] for better visualization
    
    3. Mode Energies (right column):
       - Bar plot of energy content for each mode

    Example metrics interpretation:
    - High eigenvalue magnitudes (≈1) in illumination but low in occlusion:
      → Object patterns are more stable under lighting changes than when blocked
    
    - Higher mode energies in pixel vs clutter:
      → Object patterns are more distinct from resolution changes than background changes
    
    - Lower temporal variation in illumination vs occlusion:
      → Object appearance changes less under lighting than when being blocked

    Note:
    - Comparing metrics across factors helps understand which conditions most affect object appearance
    - Higher mode energies indicate stronger patterns in the data
    - More temporal variation suggests less stable object recognition conditions
    """
    
    results = {}

    for factor in factors:
        # Check if the factor, segment, and object exist in the dataset
        if factor in dataset['test']['rgb'] and segment in dataset['test']['rgb'][factor] and object_name in dataset['test']['rgb'][factor][segment]:
            # Preprocess the data for the specific factor
            original_shape = dataset['test']['rgb'][factor][segment][object_name][0].shape
            data = preprocess_for_dmd(dataset, factor, segment, object_name)

            # Perform DMD analysis
            modes, eigenvalues, dynamics = analyze_and_plot_environmental_factor(data, factor, original_shape, r=r, plot=False)

            # Calculate metrics
            eigenvalue_magnitudes = np.abs(eigenvalues)
            mode_energies = np.sum(np.abs(modes), axis=0)
            temporal_variation = np.std(np.abs(dynamics), axis=1)

            # Store results for this factor
            results[factor] = {
                'eigenvalue_magnitudes': eigenvalue_magnitudes,
                'mode_energies': mode_energies,
                'temporal_variation': temporal_variation
            }

            # Visualization for the current factor
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))

            # Plot original first frame
            first_frame = dataset['test']['rgb'][factor][segment][object_name][0]
            axs[0].imshow(first_frame / 255.0)  # Normalize for display
            axs[0].set_title(f'{factor.capitalize()} - Original Frame')
            axs[0].axis('off')

            # Plot first DMD mode
            first_mode = np.abs(modes[:, 0])
            pixels = np.prod(original_shape)  # Total number of pixels in the RGB image
            if first_mode.size == pixels:
                first_mode = first_mode.reshape(original_shape)  # Reshape into (height, width, 3)
            else:
                first_mode = first_mode.reshape(original_shape[:2])  # Reshape into (height, width) if already collapsed
            first_mode_normalized = (first_mode - first_mode.min()) / (first_mode.max() - first_mode.min())
            im = axs[1].imshow(first_mode_normalized, cmap='viridis')
            axs[1].set_title(f'{factor.capitalize()} - First DMD Mode')
            axs[1].axis('off')
            plt.colorbar(im, ax=axs[1])

            # Plot mode energies
            axs[2].bar(range(1, r + 1), mode_energies, color='blue', alpha=0.7)
            axs[2].set_title(f'{factor.capitalize()} - Mode Energies')
            axs[2].set_xlabel('Modes')
            axs[2].set_ylabel('Energy')

            plt.tight_layout()
            plt.show()

    return results

def analyze_reconstruction_quality(original_data, modes, dynamics, original_shape, r=2):
    """
    Analyze how well DMD reconstructs the original data.

    Parameters:
    -----------
    original_data (np.array): Original video sequence, shape (n_features, n_frames)
        where n_features = height * width * 3 (for RGB channels)
        and n_frames is the number of frames
    modes (np.array): DMD modes, shape (n_features, n_modes)
        Each column represents a spatial pattern
    dynamics (np.array): Mode dynamics, shape (n_modes, n_frames-1)
        Each row shows how a mode evolves over time
    original_shape (tuple): Shape of original frames (height, width, channels)
        Used to reshape flattened data back to image format
    r (int): Number of modes to use for reconstruction
        Smaller r: simpler reconstruction, might miss details
        Larger r: more detailed reconstruction, might include noise

    Returns:
    --------
    dict: Quality metrics and reconstructed data
        'reconstruction': np.array, shape (n_features, n_frames-1)
            The reconstructed video sequence using r modes
            Can be reshaped to (height, width, 3) for each frame
        
        'error': np.array, shape (n_frames-1,)
            Frobenius norm of difference between original and reconstruction
            for each frame. Higher values indicate worse reconstruction
        
        'relative_error': np.array, shape (n_frames-1,)
            Error normalized by the norm of original data
            Values between 0 and 1:
            - Close to 0: excellent reconstruction
            - Close to 1: poor reconstruction

    Visualization:
    -------------
    The function creates a figure with two subplots:
    1. Reconstruction Error Map (left):
       - Shows spatial distribution of errors in first frame
       - Brighter areas indicate larger reconstruction errors
       - Darker areas indicate better reconstruction
       - Colorbar shows error magnitude
    
    2. Error Over Time (right):
       - Shows how reconstruction quality varies across frames
       - X-axis: Frame number
       - Y-axis: Relative error magnitude
       - Helps identify frames with poor reconstruction

    Example interpretation:
    ----------------------
    - Low average relative error (<0.1): Good reconstruction
    - High error in specific regions: Local features not well captured
    - Increasing error over time: Temporal dynamics not well captured
    - Uniform error map: Consistent reconstruction quality
    - Patchy error map: Some features harder to reconstruct

    Note:
    -----
    - The first frame is used for error map visualization
    - Reconstruction starts from second frame (due to DMD algorithm)
    - Error metrics use Frobenius norm (root sum of squared differences)
    """

    # Reconstruct the data using r modes
    reconstruction = (modes[:, :r] @ dynamics[:r, :]).real  # Reconstructed data using r modes

    # Compute reconstruction errors
    error = np.linalg.norm(original_data[:, 1:] - reconstruction, axis=0)  # Frobenius norm for each frame
    original_norms = np.linalg.norm(original_data[:, 1:], axis=0)
    relative_error = error / original_norms  # Normalize errors by original data norms

    # Visualization

    # Reconstruction error map (for the first frame)
    first_frame_error = np.abs(original_data[:, 1] - reconstruction[:, 0])  # Error for the first reconstructed frame
    error_map = first_frame_error.reshape(original_shape)  # Reshape to original frame dimensions

    # Plot error map
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    im = axs[0].imshow(error_map / error_map.max(), cmap='hot')  # Normalize error map for display
    axs[0].set_title('Reconstruction Error Map (First Frame)')
    axs[0].axis('off')
    plt.colorbar(im, ax=axs[0], orientation='vertical')

    # Error over time
    axs[1].plot(range(1, original_data.shape[1]), relative_error, marker='o', color='blue')
    axs[1].set_xlabel('Frame Number')
    axs[1].set_ylabel('Relative Error')
    axs[1].set_title('Reconstruction Error Over Time')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    # Return results
    return {
        'reconstruction': reconstruction,
        'error': error,
        'relative_error': relative_error
    }

if __name__ == "__main__":
    # Load data
    data_path = "."
    max_images = 4
    dataset = load_openloris_data(data_path, max_images_per_object=max_images)
    
    # Choose a sequence to analyze
    factor = 'illumination'
    segment = 'segment1'
    object_name = 'bottle_01'
    
    # Get original shape
    original_shape = dataset['test']['rgb'][factor][segment][object_name][0].shape
    
    # Preprocess data
    preprocessed_data = preprocess_for_dmd(dataset, factor, segment, object_name)
    
    # Perform DMD analysis
    modes, _, dynamics = analyze_and_plot_environmental_factor(
        preprocessed_data, 
        factor, 
        original_shape,
        r=5,
        plot=False
    )
    
    # Analyze reconstruction quality
    results = analyze_reconstruction_quality(
        preprocessed_data, 
        modes, 
        dynamics, 
        original_shape,
        r=2
    )
    
    # Print summary statistics
    print("\nReconstruction Quality Summary:")
    print(f"Average relative error: {np.mean(results['relative_error']):.3f}")
    print(f"Maximum relative error: {np.max(results['relative_error']):.3f}")
