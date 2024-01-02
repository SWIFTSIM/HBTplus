#!/bin/bash

# Check if we can run python3 and pip
python3 -m pip --version > /dev/null
if [[ $? -ne 0 ]]
then
  echo "ERROR: cannot run 'python3 -m pip'"
  exit 1
fi

# Check if the virtual environment with the clang formatter exists
if [ ! -d clang_formatting_env ]
then
  echo "Formatting environment not found, installing it..."
  python3 -m venv clang_formatting_env
  ./clang_formatting_env/bin/python3 -m pip install --upgrade pip # Need to access wheel  
  ./clang_formatting_env/bin/python3 -m pip install clang-format
fi


# Formatting command
clang=${CLANG_FORMAT_CMD:="clang-format"}
cmd="$clang -style=file $(git ls-files '*.c' '*.h' '*.cpp')"

# Test if `clang-format` works
command -v $clang > /dev/null
if [[ $? -ne 0 ]]
then
    echo "ERROR: cannot find $clang"
    exit 1
fi

# Print the help
function show_help {
    echo -e "This script formats HBTPlus according to Google style"
    echo -e "  -h, --help \t Show this help"
    echo -e "  -t, --test \t Test if HBTPlus is well formatted"
    echo -e "  -c, --clean \t Remove formatting venv"
}

# Remove the virtual enviroment
function remove_clang_venv {
    echo "Removing python venv used to format HBTPlus"
    rm -rf clang_formatting_env
}

# Parse arguments (based on https://stackoverflow.com/questions/192249)
TEST=0
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
	# print the help and exit
	-h|--help)
	    show_help
	    exit
	    ;;
	# check if the code is well formatted
	-t|--test)
	    TEST=1
	    shift
	    ;;
	# Remove the formatting venv
	-c|--clean)
	    TEST=1
	    remove_clang_venv
        exit
	    ;;
	# unknown option
	*)
	    echo "Argument '$1' not implemented"
	    show_help
	    exit
	    ;;
    esac
done


# Run the required commands
if [[ $TEST -eq 1 ]]
then
    # Note trapping the exit status from both commands in the pipe. Also note
    # do not use -q in grep as that closes the pipe on first match and we get
    # a SIGPIPE error.
    echo "Testing if HBTPlus is correctly formatted"
    $cmd -output-replacements-xml | grep "<replacement " > /dev/null
    status=("${PIPESTATUS[@]}")

    #  Trap if first command failed. Note 141 is SIGPIPE, that happens when no
    #  output
    if [[ ${status[0]} -ne 0 ]]
    then
       echo "ERROR: $clang command failed"
       exit 1
    fi

    # Check formatting
    if [[ ${status[1]} -eq 0 ]]
    then
        echo "ERROR: needs formatting"
        exit 1
    else
        echo "...is correctly formatted"
    fi
else
    echo "Formatting HBTPlus"
    $cmd -i
fi


