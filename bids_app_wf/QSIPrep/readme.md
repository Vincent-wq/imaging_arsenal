# Simple guide for QSIprep scripts
This is the [QSIprep](https://qsiprep.readthedocs.io/en/latest/index.html) scripts with slurm.

## General information

## Other issues

## Reminder
  1. If there are any running errors (indicated from slurm logs, ***.err), try to rerun the preprocessing for the single subject with more computing resources (cores/RAM/computing time), it works for most of time, if not, you need to check the data quality;
  2. Usually, the computing node does not have access to the Internet for security reasons, and make sure you have downloaded the templates with TemplateFlow before submitting the computing tasks.