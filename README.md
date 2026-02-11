Lazyllama is a straightforward, dependable command-line utility designed to streamline the process of supervised fine-tuning (SFT) using LoRA adapters with the LLaMA-Factory framework. Its core purpose is to make the fine-tuning experience less cumbersome and consistently repeatable.

### What Lazyllama Handles

Lazyllama manages the complete LoRA SFT workflow from start to finish:

*   **Environment Setup:** It creates and configures a Python virtual environment with all necessary dependencies.
*   **LLaMA-Factory Integration:** It clones and installs the LLaMA-Factory repository.
*   **Dataset Management:** It thoroughly inspects, validates, and provides previews of your datasets.
*   **Automatic Data Conversion:** It seamlessly converts datasets between Alpaca and ShareGPT formats.
*   **Configuration Generation:** It automatically creates YAML configuration files for both training and merging.
*   **SFT Training Execution:** It runs the supervised fine-tuning process.
*   **Safe Adapter Merging:** It handles the merging of LoRA adapters securely.
*   **Model Verification:** It performs a basic test on the merged model to ensure functionality.
*   **Resumable Operations:** It utilizes a state machine to allow for easy resumption, re-running, or cleanup of tasks.

All of this is managed through a single profile file and a single command.

### Guiding Principles

Lazyllama is built upon a few key principles:

*   **Prioritizing Reproducibility:** Every operation is logged and can be resumed, ensuring consistent results.
*   **Preventing Errors:** It offers protective measures and clear error messages rather than cryptic tracebacks.
*   **Intelligent Dataset Handling:** It automatically identifies and handles different dataset formats like Alpaca and ShareGPT.
*   **Resource Efficiency:** Default settings are optimized to minimize the risk of out-of-memory errors, especially on systems with limited VRAM.
*   **User-Friendly Experience:** It provides clear instructions, readable output, and avoids unnecessary clutter.

### System Requirements

To use Lazyllama, you'll need:

*   Python version 3.10 or higher.
*   The `git` command-line tool.
*   A Linux, WSL2, or macOS environment.
*   A CUDA-enabled GPU is recommended but not strictly required.

Python dependencies will be automatically installed within the virtual environment.

### Getting Started

Lazyllama is currently distributed as a single script within its repository. Future packaging for installation via `pip` is planned.

**Quick Start Guide:**

1.  **Initialize a Profile:**
    ```shell
    python3 lazyllama.py init
    ```
    This command generates a `lazyllama.json` file in your current directory. This file serves as the central configuration, storing details such as your chosen model, dataset information, training parameters, and output locations. It acts as the definitive record for all subsequent operations.

2.  **Execute the Full Workflow:**
    ```bash
    python3 lazyllama.py run
    ```
    When you run this command, Lazyllama will automatically perform the following steps:
    *   Create a virtual environment.
    *   Download and set up LLaMA-Factory.
    *   Prepare and validate your dataset.
    *   Generate the necessary YAML configuration files.
    *   Initiate LoRA training.
    *   Merge the trained adapters.
    *   Conduct a basic test on the resulting model.

    Any steps that have already been completed will be automatically skipped.


Lazyllama streamlines your fine-tuning workflow by meticulously managing its state. A built-in state machine keeps a detailed log of every execution step, from setting up your virtual environment and cloning repositories to training and merging models. This granular tracking means you can confidently pick up where you left off, re-run specific stages, or perform cleanups without fear of accidental re-training or re-merging.

Before diving into resource-intensive GPU tasks, Lazyllama's "Doctor" feature acts as a preflight check. It verifies essential components like your Python version, Git availability, GPU recognition (via `nvidia-smi`), disk space, virtual environment status, and the presence of the `llamafactory-cli` and Torch CUDA within your environment. The results are presented in a clear OK/WARN/FAIL table, helping you avoid wasted GPU hours.

Handling datasets is also a breeze with Lazyllama's dedicated tools. You can easily inspect your data to identify its format (JSON vs. JSONL, Alpaca vs. ShareGPT), count records, and examine its structure. Previewing random samples in a readable table or validating the dataset for missing fields and correct formatting is straightforward. Lazyllama can also convert datasets between formats, automatically detecting the current type and only converting when necessary, providing clear explanations if issues arise. For on-the-fly adjustments, you can force dataset conversion for a specific run without altering your original files.

When it's time to tidy up, Lazyllama offers a comprehensive `clean` command, or you can selectively remove configurations, previous runs, or even trained model outputs.

**Important Safety Considerations:**

*   **Never merge LoRA adapters into quantized base models.**
*   For GPUs with limited VRAM, consider setting `export_device=cpu`.
*   If you encounter Out-of-Memory (OOM) errors, try reducing `cutoff_len`, lowering `lora_rank`, or using a smaller base model. Lazyllama will alert you to potential issues but relies on your judgment for final decisions.

The name "Lazyllama" reflects the project's aim to eliminate the tedious, repetitive setup often associated with fine-tuning, freeing up your cognitive resources.



### State Management and Reproducibility

Lazyllama streamlines your fine-tuning process by meticulously tracking each execution step using a state machine. This ensures you can confidently resume, re-run, or clean up your fine-tuning jobs.

**Reproducibility with State Machine**

Lazyllama monitors every stage of your run, from setting up the virtual environment (`venv`) and cloning repositories to installing dependencies, preparing your dataset, registering configurations, training, merging, and finally, running a smoketest.

You can easily view the current status of your fine-tuning process by running:
```sell
python3 lazyllama.py state
```

To re-run from a specific point, simply specify the step:
```shell
python3 lazyllama.py rerun dataset
python3 lazyllama.py rerun train
python3 lazyllama.py rerun merge
```

This feature prevents accidental re-training or re-merging of your models.

### Preflight Checks with Doctor

Before you commit valuable GPU resources, Lazyllama's `doctor` command performs essential preflight checks:
```shell
python3 lazyllama.py doctor
```

It verifies:
*   Python version compatibility
*   Git availability
*   GPU visibility (via `nvidia-smi`)
*   Sufficient disk space
*   Existence of your virtual environment (`venv`)
*   Installation of `llamafactory-cli`
*   Torch CUDA availability within your `venv`

The output is presented in a clear table indicating `OK`, `WARN`, or `FAIL` for each check.

### Advanced Dataset Tools

Lazyllama offers robust tools for managing your datasets:

*   **Inspect a dataset:**
    ```shell
    python3 lazyllama.py dataset inspect data.jsonl
    ```
    This command detects the dataset format (JSON vs. JSONL, Alpaca vs. ShareGPT), counts the records, and analyzes the key structure.

*   **Preview samples:**
    ```shell
    python3 lazyllama.py dataset preview data.jsonl -n 3
    ```
    Get a readable table of randomly selected samples from your dataset.
    

*   **Validate dataset format:**
    ```shell
    python3 lazyllama.py dataset validate data.jsonl
    ```
    This validation will clearly fail if required fields are missing or if the dataset doesn't conform to either the Alpaca or ShareGPT format.

*   **Convert datasets:**
     ```shell
     python3 lazyllama.py dataset convert data.jsonl --to alpaca --out data_alpaca.jsonl
     python3 lazyllama.py dataset convert data.jsonl --to sharegpt --out data_sharegpt.jsonl
    ```
    Lazyllama automatically detects the current format, converts only when necessary, and provides explanations if conversion fails.

### Runtime Dataset Conversion**

For a single run, you can force dataset conversion without altering your original files:
```shell
python3 lazyllama.py run --convert-to alpaca
python3 lazyllama.py run --convert-to sharegpt
```

### Effortless Cleaning

Maintain a tidy workspace with the `clean` command:
```shell
python3 lazyllama.py clean
```

You can also selectively clean specific components:
```shell
python3 lazyllama.py clean --configs
python3 lazyllama.py clean --runs
python3 lazyllama.py clean --outputs (Note: This will delete trained models.)
```

**Important Safety Considerations**

*   **Never merge LoRA adapters into quantized base models.**
*   For GPUs with limited VRAM, consider using `export_device=cpu`.
*   If you encounter Out-Of-Memory (OOM) errors, try reducing `cutoff_len`, lowering `lora_rank`, or using a smaller base model.

While Lazyllama provides warnings, it relies on you to make the final decisions.

The name "Lazyllama" reflects a philosophy centered on:

*   Avoiding redundant effort in fine-tuning configurations.
*   Prioritizing error prevention over post-hoc debugging.

### Future developments are anticipated to include:

*   A readily installable package via `pip`.
*   Automatic VRAM-based tuning.

