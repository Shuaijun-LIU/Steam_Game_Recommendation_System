## Web Scraping with Selenium

This guide will walk you through the steps required to set up your environment for running a Selenium-based web scraping script. You will learn how to install necessary Python packages and configure Chromedriver on different operating systems.

### Prerequisites

Before you begin, ensure you have Python installed on your system. Python 3.6 or newer is recommended.

### Installing Required Python Packages

The script requires several Python packages including `selenium`, `pandas`, and `urllib`. You can install these packages using pip. Open your command line (Terminal for Mac and Linux, Command Prompt or PowerShell for Windows) and run the following command:

```bash
pip install selenium pandas
```

### Setting Up Chromedriver

Selenium requires a driver to interface with the chosen browser. Chromedriver is used for Google Chrome. Follow the steps below based on your operating system to download and set up Chromedriver.

#### Windows

1. Download Chromedriver:
   - Go to the [Chromedriver download page](https://sites.google.com/chromium.org/driver/).
   - Choose the version that matches your Chrome browser's version and download the Windows zip file.
2. Unzip the downloaded file to a known directory, for example, `C:\chromedriver\`.
3. Add the path to Chromedriver (`C:\chromedriver\`) to your system's Environment Variables:
   - Right-click on 'This PC' or 'My Computer' and select 'Properties'.
   - Click on 'Advanced system settings'.
   - In the System Properties window, click on the 'Environment Variables' button.
   - In the Environment Variables window, select 'Path' under 'System variables' and click 'Edit'.
   - Click 'New' and add the path to where Chromedriver is located (`C:\chromedriver\`).
   - Click OK to close all dialogs.

#### Mac

1. Download Chromedriver:
   - Go to the [Chromedriver download page](https://sites.google.com/chromium.org/driver/).
   - Choose the version that matches your Chrome browser's version and download the Mac zip file.
2. Unzip the downloaded file and move `chromedriver` to the `/usr/local/bin/` directory using the Terminal:
   ```bash
   mv path/to/chromedriver /usr/local/bin/
   ```
3. Verify that Chromedriver is set up correctly by running:
   ```bash
   chromedriver --version
   ```

#### Linux

1. Download Chromedriver:
   - Go to the [Chromedriver download page](https://sites.google.com/chromium.org/driver/).
   - Choose the version that matches your Chrome browser's version and download the Linux zip file.
2. Unzip the downloaded file and move `chromedriver` to the `/usr/local/bin/` directory using the Terminal:
   ```bash
   sudo mv path/to/chromedriver /usr/local/bin/
   sudo chmod +x /usr/local/bin/chromedriver
   ```
3. Verify that Chromedriver is set up correctly by running:
   ```bash
   chromedriver --version
   ```

### Running the Script

Once you have all the prerequisites installed and Chromedriver configured, you can run the script by navigating to the directory containing the script and running:

```bash
python script_name.py
```

Replace `script_name.py` with the name of your Python script.

### Troubleshooting

If you encounter issues, ensure that your version of Chromedriver matches the version of Google Chrome installed on your system. Also, check that the Python packages are installed correctly and that the PATH variable includes Chromedriver.

### References

For further reading and resources, consider checking the following links:

- [Selenium Documentation](https://selenium-python.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Chromedriver Downloads](https://sites.google.com/chromium.org/driver/)
