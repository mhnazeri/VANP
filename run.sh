#! /usr/bin/env bash

export PROJECT_NAME=VANP  # add your project folder to python path
export LOCAL_RANK=1
export NUM_TRAINERS=2 # number of GPUs you have
export PYTHONPATH=$PYTHONPATH:$PROJECT_NAME
export COMET_LOGGING_CONSOLE=info

Help()
{
   # Display Help
   echo 
   echo "Facilitates running different stages of training and evaluation."
   echo 
   echo "options:"
   echo "train                      Trains the model."
   echo "parser                     Parse bag files."
   echo "run file_name              Runs file_name.py file."
   echo
}

run () {
  case $1 in

    train)
      if [[ -z $2 ]]
      then
        python $PROJECT_NAME/train_pretext.py --conf $PROJECT_NAME/conf/config_pretext
      else
        echo "Using provided config file {$2} for pre-training."
        python $PROJECT_NAME/train_pretext.py --conf $2
      fi
      ;;
    parser)

      if [[ $2 == 'sampler' ]]
      then
        echo "Creating samples from bag files!"
        python $PROJECT_NAME/utils/parser.py --conf $PROJECT_NAME/conf/parser -cs
      else
        echo "Parsing $2 bags."
        python $PROJECT_NAME/utils/parser.py --name $2 --conf $PROJECT_NAME/conf/parser

      fi
      ;;
    run)
      python $2
      ;;
    -h) # display Help
      Help
      exit
      ;;
    *)
      echo "Unknown '$1' argument. Please run with '-h' argument to see more details."
      # Help
      exit
      ;;
  esac
}

run $1 $2 $3

# echo "Done."
