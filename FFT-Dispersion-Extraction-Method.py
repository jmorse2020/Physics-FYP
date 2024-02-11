import numpy as np
import os
import matplotlib.pyplot as plt
import mplcursors
mplcursors.cursor(hover=True)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def display_text_with_border(message, color=None, clear = False):
    if clear:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear the terminal screen


    border = '*' * (len(message) + 4)
    formatted_message = f"\n{border}\n* {message} *\n{border}\n"

    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'reset': '\033[0m'
    }

    if color and color in colors:
        print(colors[color] + formatted_message + colors['reset'])
    else:
        print(formatted_message)

def display_text(message, color=None):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'reset': '\033[0m'
    }

    if color and color in colors:
        print(colors[color] + message + colors['reset'])
    else:
        print(message)

def read_settings(settings_file_path):
    import json
    global settings
    try:
        with open(settings_file_path, 'r') as file:
            settings = json.load(file)
            # display_text("Settings loaded successfully", color='green')
        return settings
    except FileNotFoundError:
        display_text(f"Error: Settings file not found at '{settings_file_path}'.", color='red')
        exit()
    except json.JSONDecodeError:
        display_text(f"Error: Unable to decode JSON in the settings file at '{settings_file_path}'. Please check the file format.", color = 'red')
        exit()
    except Exception as e:
        display_text(f"An unexpected error occurred: {e}", color='red')
        exit()

settings_file = "settings.json"
clear_screen()
display_text("Importing modules ...", color='green')
try:
    from Functions import DispersionExtraction as de
except ImportError:
    display_text("Error: The required module 'Functions.DispersionExctraction' could not be imported. Please ensure the file is in the same folder as this script.", color='red')
    exit()
display_text(f"Reading '{settings_file}' ...", color='green')
settings = read_settings(settings_file)

def count_csv_files(directory_path):
    try:
        csv_files = [file for file in os.listdir(directory_path) if file.lower().endswith(".csv")]
        csv_count = len(csv_files)
        return csv_count, csv_files
    except FileNotFoundError:
        return -1, []
    except Exception as e:
        display_text(f"An unexpected error occurred: {e}", color='red')


def count_files_in_directory(directory_path):
    try:
        file_count = len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])
        for f in os.listdir(directory_path):
            if os.path.isfile(os.path.join(directory_path, f)):
                print(f)
        return file_count
    except FileNotFoundError:
        return -1  # Indicates that the directory doesn't exist
    except Exception as e:
        display_text(f"An unexpected error occurred: {e}", color='red')
        return -1

def verify_and_read_OSA_files_from_directory(save_outputs_path, settings, save_filename = ""):
    ref_count = 0
    fib_count = 0
    int_count = 0
    csv_count, csv_files = count_csv_files(save_outputs_path)
    if csv_count != 3:
        display_text(f"CSV Count: {count_csv_files(save_outputs_path)}", color='red')
        return False
    for file in csv_files:
        if os.path.isfile(os.path.join(save_outputs_path, file)):
            file_path = os.path.join(save_outputs_path, file)
            if file.lower().__contains__("ref") and ref_count == 0: 
                print(file_path)
                [reference_wavelengths, reference_intensity] = read_OSA_trace(file_path, settings)
                reference_intensity = np.log10(reference_intensity)
            elif file.lower().__contains__("fib") and fib_count == 0:
                [fibre_wavelengths, fibre_intensity] = read_OSA_trace(file_path, settings)
                fibre_intensity = np.log10(fibre_intensity)
            elif file.lower().__contains__("int") and int_count == 0:
                [interference_wavelengths, interference_intensity] = read_OSA_trace(file_path, settings)
                interference_intensity = np.log10(interference_intensity)
            else:
                display_text(f"File {file} unable to be read", color='red')
                return False
    arrays_equal = np.array_equal(reference_wavelengths, fibre_wavelengths) and np.array_equal(fibre_wavelengths, interference_wavelengths)
    if not arrays_equal:
        display_text("Wavelength arrays of each trace are not equal. Attempting to proceed with reference trace wavelengths ...", color='red')
    save_outputs_path = create_output_folder_with_foldername(save_outputs_path, save_filename)
    output = restrict_OSA_trace_domain(reference_wavelengths, reference_intensity, fibre_intensity, interference_intensity, save_outputs_path)    
    reference_wavelengths = output[0]
    reference_intensity = output[1]
    fibre_intensity = output[2]
    interference_intensity = output[3]
    scaled_interference_intensity = extract_scaled_interference_intensity(reference_intensity, fibre_intensity, interference_intensity)
    plt.close()
    plt.ion()
    # *** UNDO THIS IF YOU WANT TO WORK OUT THE LOG(0) ERROR ... *** #
    # if settings["OSA_file_is_log"]:
    #     plt.plot(reference_wavelengths, np.log(scaled_interference_intensity), label = "Scaled interference", linewidth=0.8, color='green')
    # else:
    #     plt.plot(reference_wavelengths, scaled_interference_intensity, label = "Scaled interference", linewidth=0.8, color='green')
    plt.plot(reference_wavelengths, scaled_interference_intensity, label = "Scaled interference", linewidth=0.8, color='green')
    plt.xlabel("Wavelengths [nm]")
    plt.ylabel("Intensity")
    plt.savefig(os.path.join(save_outputs_path, "scaled_interference.png"), dpi = 1000)
    # Save data
    plt.show()
    # display_text("Traces loaded successfully ...", color='green')
    return [reference_wavelengths, scaled_interference_intensity, save_outputs_path]

def get_valid_input(prompt, lower_limit, upper_limit):
    while True:
        user_input = input(prompt)
        try:
            # Attempt to convert the input to a numeric type
            user_input = float(user_input)

            # Check if the input is within the specified range
            if lower_limit <= user_input <= upper_limit:
                return user_input
            else:
                display_text(f"Input must be between {lower_limit} and {upper_limit}. Try again.", color='red')
        except ValueError:
            display_text("Invalid input. Please enter a numeric value.", color='red')

    
def restrict_OSA_trace_domain(reference_wavelengths, reference_intensity, fibre_intensity, interference_intensity, save_outputs_path):
    import matplotlib.pyplot as plt
    display_text("Traces plotted ...", color='blue')
    plot_all_traces(reference_wavelengths, reference_intensity, fibre_intensity, interference_intensity, save_outputs_path, show_cutoff_lines = True)
    while True:
        YN = input("Do you wish to change the cut-off wavelengths? (Y/N): ")
        if YN.lower() in ['y', 'yes']:
            try:
                global settings
                print(f"Settings before: min {settings['OSA_file_keep_min_wavelength']}, max {settings['OSA_file_keep_max_wavelength']}")
                new_min_wavelength = get_valid_input("Enter the minimum wavelength: ", min(reference_wavelengths), max(reference_wavelengths))
                update_JSON("OSA_file_keep_min_wavelength", new_min_wavelength, settings_file)
                new_max_wavelength = get_valid_input("Enter the maximum wavelength: ", new_min_wavelength, max(reference_wavelengths))
                update_JSON("OSA_file_keep_max_wavelength", new_max_wavelength, settings_file)
                settings = read_settings(settings_file)
                print(f"Settings after: min {settings['OSA_file_keep_min_wavelength']}, max {settings['OSA_file_keep_max_wavelength']}")
                plt.close()
                reference_output = execute_OSA_domain_restriction(reference_wavelengths, reference_intensity)                
                fibre_output = execute_OSA_domain_restriction(reference_wavelengths, fibre_intensity)                
                interference_output = execute_OSA_domain_restriction(reference_wavelengths, interference_intensity)
                # interference_intensity = output[1]
                # reference_wavelengths = output[0]
                plt.close()
                plot_all_traces(reference_wavelengths, reference_intensity, fibre_intensity, interference_intensity, save_outputs_path, show_cutoff_lines = True, save_plots = False)
                if YN.lower() in ['y', 'yes']:
                    continue
                elif YN.lower() in  ['n', 'no']:
                    interference_intensity = interference_output[1]
                    reference_wavelengths = reference_output[0]
                    reference_intensity = reference_output[1]
                    fibre_intensity = fibre_output[1]
                    plt.savefig(os.path.join(save_outputs_path, "OSA_traces.png"), dpi = 1000)                    
                    break
                else:
                    display_text("Unrecognised input. Allowing re-adjustment: ", color='red')
                    continue
            except Exception as e:
                display_text(f"An unexpected error occured: {e}", color='red')
                continue
            break
        elif YN.lower() in ['n', 'no']:
            plt.close()
            output = execute_OSA_domain_restriction(reference_wavelengths, reference_intensity)
            reference_intensity = output[1]
            output = execute_OSA_domain_restriction(reference_wavelengths, fibre_intensity)
            fibre_intensity = output[1]
            output = execute_OSA_domain_restriction(reference_wavelengths, interference_intensity)
            interference_intensity = output[1]
            reference_wavelengths = output[0]
            plot_all_traces(reference_wavelengths, reference_intensity, fibre_intensity, interference_intensity, save_outputs_path, show_cutoff_lines = True, save_plots = True)
            break
        else:
            display_text("Unrecognised input, please enter 'y' or 'n'.", color='red')
    return [reference_wavelengths, reference_intensity, fibre_intensity, interference_intensity]



def rolling_average(arr, window_size):
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array to consider every window of size window_size
    while i < len(arr):
        # Determine the start and end indices of the current window
        start_index = max(0, i - window_size + 1)
        end_index = i + 1

        # Extract elements within the current window
        window = arr[start_index:end_index]

        # Calculate the average of the current window
        window_average = sum(window) / len(window)

        # Store the average of the current window in the moving averages list
        moving_averages.append(window_average)

        # Move to the next position
        i += 1

    return moving_averages

def execute_OSA_domain_restriction(reference_wavelengths, intensity):
    min_spectral_phase_wavelength = settings["OSA_file_keep_min_wavelength"]
    max_spectral_phase_wavelength = settings["OSA_file_keep_max_wavelength"]
    if min_spectral_phase_wavelength != None and max_spectral_phase_wavelength != None:
        indices = np.where((reference_wavelengths >= min_spectral_phase_wavelength) & (reference_wavelengths <= max_spectral_phase_wavelength))[0]
        reference_wavelengths = reference_wavelengths[indices]
        intensity = intensity[indices] 
    return [reference_wavelengths, intensity]
    
def create_output_folder_with_foldername(directory, folder_name):
    # Get the directory and filename from the provided file path
    # directory, filename = os.path.split(file_path)

    # Extract the file name without extension
    # file_name, file_extension = os.path.splitext(filename)

    # Define the base output folder name
    if folder_name == "":
        output_folder_base = "out"
    else:
        output_folder_base = folder_name + '_out'
    
    # Initialize a counter to handle folder name increments
    counter = 1

    while True:
        # Form the output folder name with an optional counter
        output_folder_name = f"{output_folder_base}_{counter}" if counter > 1 else output_folder_base

        # Combine the output folder name with the original directory
        output_folder_path = os.path.join(directory, output_folder_name)

        # Check if the folder already exists
        if not os.path.exists(output_folder_path):
            # Create the output folder
            os.makedirs(output_folder_path)
            display_text(f"Output folder '{output_folder_name}' created.", color='green')
            return output_folder_path
        else:
            # Increment the counter if the folder already exists
            counter += 1

# # Example usage:
# file_path = input("Enter the file path: ")
# output_folder = create_output_folder(file_path)
# print(f"Output folder path: {output_folder}")

def display_menu(options):
    # clear_screen()
    print("MENU:")
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")

def plot_all_traces(wavelengths, reference_intensity, fibre_intensity, interference_intensity, save_outputs_path, show_cutoff_lines= False, save_plots = False, log = False, linewidth=0.8):
    import matplotlib.pyplot as plt
    plt.ion()
    if log:
        plt.plot(wavelengths, np.log10(reference_intensity), label = "Reference", linewidth=linewidth)
        plt.plot(wavelengths, np.log10(fibre_intensity), label = "Fibre", linewidth=linewidth)
        plt.plot(wavelengths, np.log10(interference_intensity), label = "Interference", linewidth=linewidth)
    else:
        plt.plot(wavelengths, reference_intensity, label = "Reference", linewidth=linewidth)
        plt.plot(wavelengths, fibre_intensity, label = "Fibre", linewidth=linewidth)
        plt.plot(wavelengths, interference_intensity, label = "Interference", linewidth=linewidth)
    if show_cutoff_lines:
        plt.axvline(x = settings["OSA_file_keep_min_wavelength"], color = 'red', linestyle = '--', label = f"x = {settings['OSA_file_keep_min_wavelength']}")
        plt.axvline(x = settings["OSA_file_keep_max_wavelength"], color = 'red', linestyle = '--', label = f"x = {settings['OSA_file_keep_max_wavelength']}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    # plt.ylim([min(reference_intensity), max(np.array(rolling_average(reference_intensity, 5)))])
    plt.legend()
    plt.show()
    # input("Press any key to close figure")
    # plt.close()
    if save_plots:
        plt.savefig(os.path.join(save_outputs_path, "OSA_traces.png"), dpi = 1000)
        display_text("OSA traces saved", color='green')
    # plt.ioff()

def execute_option(option, settings):
    # clear_screen()
    if option == 1:
        display_text("Under construction ...", color='red')
        return
        # save_outputs_path = ""
        # reference_wavelengths = ""
        # scaled_interference_intensity = ""
        # display_text("You selected option 1: Manual Mode")
        # display_text("Please enter the paths to each of the three traces: ")
        # path_to_reference_OSA_trace = get_user_file_path("\nReference arm trace: ")
        # [reference_wavelengths, reference_intensity] = read_OSA_trace(path_to_reference_OSA_trace, settings)
        # path_to_fibre_OSA_trace = get_user_file_path("Fibre arm trace: ")
        # [fibre_wavelengths, fibre_intensity] = read_OSA_trace(path_to_fibre_OSA_trace, settings)
        # path_to_interference_OSA_trace = get_user_file_path("Interference trace: ")
        # [interference_wavelengths, interference_intensity] = read_OSA_trace(path_to_interference_OSA_trace, settings)

        # arrays_equal = np.array_equal(reference_wavelengths, fibre_wavelengths) and np.array_equal(fibre_wavelengths, interference_wavelengths)
        # if not arrays_equal:
        #     display_text("Wavelength arrays of each trace are not equal. Attempting to proceed with reference trace wavelengths ...", color='red')
        # scaled_interference_intensity = extract_scaled_interference_intensity(reference_intensity, fibre_intensity, interference_intensity)
        # # display_text("Traces loaded successfully ...", color='green')

        # save_outputs_path = ""
        # while True:
        #     save_outputs_path = input("Please enter a save location: ")
        #     if is_valid_directory(save_outputs_path):
        #         folder_name = str(input("Please enter a folder name: "))
        #         # save_outputs_path = os.path.join(save_outputs_path, folder_name)
        #         save_outputs_path = create_output_folder_with_foldername(save_outputs_path, folder_name)
        #         break
        #     else:
        #         display_text("Not a valid save location.", color='red')
        #     return [save_outputs_path, reference_wavelengths, scaled_interference_intensity]
    elif option == 2:
        save_outputs_path = ""
        reference_wavelengths = ""
        scaled_interference_intensity = ""
        print("You selected option 2: Automated")
        input("You must have the three traces only in a folder. Each must contain the key 'ref', 'fibre', 'int' accordingly. Press return key to continue")
        while True:
            save_outputs_path = input("Please enter the full path to directory containing 3 OSA traces: ")
            interference_trace = verify_and_read_OSA_files_from_directory(save_outputs_path, settings)
            if is_valid_directory(save_outputs_path) and interference_trace is not False:
                # create_output_folder_with_foldername(save_outputs_path, "")
                reference_wavelengths = np.array(interference_trace[0])
                scaled_interference_intensity = np.array(interference_trace[1])
                save_outputs_path = interference_trace[2]
                break
            else:
                display_text("Either the directory doesn't exist or files not in correct form.", color='red')
        return [save_outputs_path, reference_wavelengths, scaled_interference_intensity]
    elif option == 3:
        display_text("Under construction ...", color='red')
        # save_outputs_path = ""
        # reference_wavelengths = ""
        # scaled_interference_intensity = ""
        # print("You selected option 3. Execute corresponding action.")
        # Add your action for option 3 here
    elif option == 4:
        display_text("Exiting ...", color='blue')
        exit()
    else:
        print("Invalid option. Please choose a valid option.")

def get_user_file_path(prompt, expected_type=str, file_must_exist=True):
    while True:
        user_input = input(prompt)
        try:
            user_input = expected_type(user_input)
            if expected_type == str and file_must_exist and not os.path.isfile(user_input):
                raise ValueError("File does not exist. Please enter a valid file path.")            
            # If the input is successfully converted and validated, break out of the loop
            display_text("File path validated", color='green')
            break
        except ValueError as e:
            display_text(f"Invalid input: {e}", color='red')
            # Prompt the user again if the input is not of the expected type or if file validation fails
    return user_input

def get_user_directory_path(prompt, file_must_exist=True):
    while True:
        user_input = input(prompt)
        try:
            # Validate that the entered path is an existing directory
            if file_must_exist and not os.path.isdir(user_input):
                raise ValueError("Directory does not exist. Please enter a valid directory path.")
            
            # If the input is successfully validated, break out of the loop
            display_text("Directory path validated", color='green')
            break
        except ValueError as e:
            display_text(f"Invalid input: {e}", color='red')
            # Prompt the user again if the input is not a valid directory
    return user_input

def _ground_and_normalise(y_data):
        y_data = y_data - min(y_data)
        return (y_data - min(y_data))/ max(y_data - min(y_data))

def extract_scaled_interference_intensity(reference_intensity, fibre_intensity, interference_intensity, ground_and_normalise = False):
    scaled = ((interference_intensity - reference_intensity - fibre_intensity) + np.sqrt(reference_intensity * fibre_intensity)) / (2 * np.sqrt(reference_intensity * fibre_intensity))
    if ground_and_normalise:
        scaled = _ground_and_normalise(scaled)
    return scaled


def LogConversion(y):
        return 10**(y)

def read_OSA_trace(path_to_OSA_trace, settings, show_plot = False):
    import pandas as pd
    try:
        # print(settings["OSA_file_header"])
        data = pd.read_csv(path_to_OSA_trace, header = settings["OSA_file_header"], skiprows = settings["OSA_file_skip_rows"])
        # Could in theory do something to detect the preamble and ensure it is in nm
        wavelengths = data[0]
        intensity = data[1]
        if show_plot:
            import matplotlib.pyplot as plt
            plt.plot(wavelengths, intensity)
            plt.show()        
        if settings["OSA_file_is_log"]:
            intensity = LogConversion(intensity)
        return [wavelengths, intensity]
    except Exception as e:
        display_text(f"An unexpected error occurred while reading the OSA trace file: {e}", color='red')

def update_JSON(key, new_value, file_path):
    import json
    global settings
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        display_text(f"Unable to read {file_path}.", color='red')
        return False

    # Step 2: Modify the data in memory (for example, adding a new key-value pair)
    data[key] = new_value

    # Step 3: Write the updated data back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2) 
    settings = read_settings(file_path)
    

def hline(length = 100, style = "-", color = None):
    line = "\n"
    for i in range(length):
        line += style
    line += "\n"
    display_text(line, color=color)

def is_valid_directory(path):
    return os.path.isdir(path)

def write_csv(file_path, data, headers, preamble = []): 
        import csv       
        if len(headers) != len(data):
            print("Enter one header per coumn.")
            return
        zipped_data = zip(*data)
        data_list = list(zipped_data)
        # check file path name ends in csv
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            if preamble is not []:
                writer.writerow(preamble)
            writer.writerow(headers)
            writer.writerows(data_list)

# *** Script Starts Here *** #
# Load tests ...

# clear_screen()
display_text_with_border("Welcome to FFT-Dispersion-Extraction-Console")
display_text("About:\nThis script allows you to display and save the dispersion characteristics of output from OSA traces.")
hline()

menu_options = ["Manual", "Automated", "Edit settings", "Exit"]

while True:
    display_menu(menu_options)
    try:
        user_input = int(input("Enter the number of your choice: "))
        if 1 <= user_input <= len(menu_options):
            [save_outputs_path, reference_wavelengths, scaled_interference_intensity] = execute_option(user_input, settings)
            break
        else:
            display_text("Invalid input. Please enter a number corresponding to the menu options.", color='red')
    except ValueError:
        display_text("Invalid input. Please enter a number.", color='red')


# *** EXTRACTION ***
display_text("Extracting ... ", color='green')
# print(reference_wavelengths)
# print(scaled_interference_intensity)
# plt.close()
# plt.plot(reference_wavelengths, scaled_interference_intensity, label="Extracted spectral phase", linewidth = 0.8, color = 'g')
# plt.xlim([650,1400])
# plt.ylim([0.95,1.1])
# plt.title("Spectral Interference (with POI)")
# plt.xlabel("Wavelengths [nm]")
# plt.ylabel("Intensity")
# plt.savefig("/Users/jackmorse/Desktop/SI-with-POI.png", dpi = 1200)
# plt.show()
# input()

[xf, yf] = de.TraceFFT(reference_wavelengths, scaled_interference_intensity, settings["normalise"], settings["hanning"])
display_text("FFT plotted ...", color='blue')
plt.close()
plt.ion()
linewidth = 0.8
plt.plot(xf, np.log10(np.abs(yf)), linewidth = linewidth)
plt.title("FFT")
plt.xlabel("Fourier domain")
plt.ylabel(r'$\log_{10}|\mathcal{F}(I)|$')
plt.axvline(settings["keep_min_freq"], color = 'red', linestyle = '--', label="Cut-off")
idx = de.FilterIndicesFFT(xf, np.abs(yf), settings["side"], settings["keep_min_freq"], settings["keep_max_freq"])
filtered_fourier_data = de.BoxFilter(yf, idx)
plt.plot(xf[idx], np.log10(np.abs(filtered_fourier_data[idx])), color='red', linewidth = linewidth)
plt.legend()
mplcursors.cursor(hover=True)
plt.show()
YN = input("Do you wish to change the FFT cut-off? (Y/N): ")
while True:
    if YN.lower() in ['y', 'yes']:
        try:
            print(f"Current cut-off value: left {settings['keep_min_freq']}, right {settings['keep_max_freq']}")
            new_cutoff = get_valid_input("Enter the new cut-off: ", 0, max(xf))
            update_JSON("keep_min_freq", new_cutoff, settings_file)
            settings = read_settings(settings_file)
            print(f"New cut-off value: left {settings['keep_min_freq']}, right {settings['keep_max_freq']}")
            plt.close()
            idx = de.FilterIndicesFFT(xf, yf, settings["side"], settings["keep_min_freq"], settings["keep_max_freq"])
            filtered_fourier_data = de.BoxFilter(yf, idx)
            plt.close()
            plt.ion()
            plt.plot(xf, np.log10(np.abs(yf)), linewidth = linewidth)
            plt.title("FFT")
            plt.xlabel("Fourier domain")
            plt.ylabel(r'$\log_{10}|\mathcal{F}(I)|$')
            plt.axvline(settings["keep_min_freq"], color = 'red', linestyle = '--', label=f"x = {settings['keep_min_freq']}")
            plt.plot(xf[idx], np.log10(np.abs(filtered_fourier_data[idx])), color='red', linewidth = linewidth)
            plt.legend()
            YN = input("Do you wish to re-adjust? (Y/N): ")
            if YN.lower() in ['y', 'yes']:
                continue
            elif YN.lower() in  ['n', 'no']:
                break
            else:
                display_text("Unrecognised input. Allowing re-adjustment: ", color='red')
                continue
        except Exception as e:
            display_text(f"An unexpected error occured: {e}", color='red')
        break
    elif YN.lower() in ['n', 'no']:
        plt.close()
        break
    else:
        display_text("Unrecognised input, please enter 'y' or 'n'.", color='red')
# plt.close()
# plt.ion()
# plt.plot(xf, np.log10(np.abs(yf)), linewidth = linewidth)
# plt.title("FFT")
# plt.xlabel("Fourier domain")
# plt.ylabel(r'$\log_{10}|\mathcal{F}(I)|$')
# # plt.axvline(settings["keep_min_freq"], color = 'red', linestyle = '--')
# plt.plot(xf[idx], np.log10(np.abs(filtered_fourier_data[idx])), color='red', linewidth = linewidth)
# input("Press any key to close the figure ...")
plt.savefig(os.path.join(save_outputs_path, "FFT.png"), dpi = 1000)
display_text("FFT saved", 'green')

filtered_y = de.InverseFFT(filtered_fourier_data)

final_ys = de.ExtractAndUnwrap(filtered_y)
display_text("Unwrap plotted ...", color='blue')
plt.close()
plt.ion()
plt.plot(reference_wavelengths, final_ys, color='green', linewidth=linewidth)
plt.xlabel("Wavelengths [nm]")
plt.ylabel("Phase [rad]")
plt.title("Unwrapped Phase")
input("Press any key to close the figure ...")
plt.savefig(os.path.join(save_outputs_path, "Unwrap.png"), dpi = 1000)
display_text("Unwrap saved", 'green')
while True:
    fibre_length = input("Enter the fibre length in meters: ")
    try:
        fibre_length = float(fibre_length)
        if 0 < fibre_length:
            break
        else:
            display_text(f"Fibre length must be greater than 0", color='red')
    except ValueError:
        display_text("Invalid input. Please enter a numeric value.", color='red')
display_text(f"Fibre length: {fibre_length} m")

beta_lambda = de.ObtainBetaFromPhi(final_ys, fibre_length)
display_text("Beta plotted ...", color='blue')
plt.close()
plt.ion()
plt.plot(reference_wavelengths, beta_lambda, color='green')
plt.title("Beta")
plt.xlabel("Wavelengths [nm]")
plt.ylabel(r'$\beta(\lambda)$')
while True:
        smooth_over = input("Enter the number of points to smooth over: ")
        try:
            smooth_over = int(smooth_over)
            beta_lambda_smooth = rolling_average(beta_lambda, smooth_over)
            plt.plot(reference_wavelengths, beta_lambda_smooth, label=f"smooth_over = {smooth_over}")
            plt.legend()
            YN = input("Is this smooth enough? (Y/N): ")
            if YN.lower() in ['y', 'yes']:
                break
            elif YN.lower() in ['n', 'no']:
                continue
            else:
                display_text("Invalid input. Taken as 'n'")
                continue
            break
        except ValueError:
            display_text("Invalid input. Please enter an integer value.", color='red')
plt.show()
input("Press any key to close the figure ...")
plt.savefig(os.path.join(save_outputs_path, "beta.png"), dpi = 1000)
display_text("Beta saved", color='green')
# print("CDA")
# plt.close()
# plt.ion()
# plt.plot(reference_wavelengths, de.CDA2(rolling_average(beta_lambda, 10), reference_wavelengths[1] - reference_wavelengths[0]))
# plt.show()
# plt.savefig(os.path.join(save_outputs_path, "smoothed_CDA1.png"), dpi = 1000)
# display_text("CDA figure saved", color='green')
# input("Press any key to close the figure ...")

refractive_index_lambda = de.ObtainRefractiveIndex(beta_lambda, reference_wavelengths)
GVD_lambda = de.GVD_lambda(beta_lambda_smooth, reference_wavelengths)
plt.close()
plt.ion()
display_text("GVD plotted ...", color='blue')
plt.plot(reference_wavelengths, GVD_lambda)
GVD_lambda_smooth = rolling_average(GVD_lambda, smooth_over)
plt.plot(reference_wavelengths, GVD_lambda_smooth, label = f"Smoothed GVD ({smooth_over})")
plt.title("Group Velocity Dispersion")
plt.xlabel("Wavelengths [nm]")
plt.ylabel("GVD [ps / (nm km)]")
plt.show()
input("Press any key to close the figure ...")
plt.savefig(os.path.join(save_outputs_path, "GVD.png"), dpi = 1000)
write_csv(os.path.join(save_outputs_path, "GVD.csv"), [reference_wavelengths, GVD_lambda, GVD_lambda_smooth], ["wavelengths[nm]", "GVD[ps_nmkm]", "smooth_GVD[ps_nmkm]"])
display_text("GVD saved", color='green')
Vg_lambda = de.Vg_lambda(beta_lambda, reference_wavelengths)
display_text_with_border("Complete!", color='green')
plt.close()
id = np.where((reference_wavelengths >= 950) & (reference_wavelengths <= 1400))[0]
print(id)
plt.plot(np.array(reference_wavelengths)[id], np.array(GVD_lambda_smooth)[id])
plt.title("GVD.......")
input("Press any key to exit ...")
plt.close()
exit()
