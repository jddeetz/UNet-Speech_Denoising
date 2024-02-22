# Use pip3 to install the required modules
pip3 install -r requirements.txt

##### MAKE WAV FILES FROM SYNTHETIC DATA
# Clone this repository, which allows us to generate audio with synthetic noise on top of it.
git clone git@github.com:microsoft/MS-SNSD.git
# Performs surgical bugfix on line 15 because the repo above has a bug. 
# Note the '' a Mac OSX option and will not work in Linux 
sed -i '' '15s/float/int/g' MS-SNSD/noisyspeech_synthesizer.py
# Makes edits to the MS-SNSD config file to generate training data
# Changes from generating 1 hour of wav files (default) to 30 minutes
sed -i '' '22s/1/0.5/g' MS-SNSD/noisyspeech_synthesizer.cfg
# Changes from generating 5 different levels of background noise to just 1 level
sed -i '' '25s/5/1/g' MS-SNSD/noisyspeech_synthesizer.cfg
# Ignores all types of background noise other than AirConditioning
sed -i '' '29s/None/AirportAnnouncements,Babble,Bus,CafeTeria,Cafe,Car,CopyMachine,Field,Hallway,Kitchen,LivingRoom,Metro,Munching,Neighbor,NeighborSpeaking,Office,Park,Restaurant,ShuttingDoor,Square,SqueakyChair,Station,Traffic,Typing,VacuumCleaner,WasherDryer,Washing/g' MS-SNSD/noisyspeech_synthesizer.cfg
# Generate training data files for the noisy speech dataset
python3 MS-SNSD/noisyspeech_synthesizer.py
# Move the data into a data directory in the main repo
mkdir data
mv MS-SNSD/CleanSpeech_training data/CleanSpeech
mv MS-SNSD/NoisySpeech_training data/NoisySpeech


##### GENERATE SPECTROGRAMS FROM WAV FILES
# Generate spectrograms in data directory
python3 generate_spectrograms.py --input_directory data/NoisySpeech --output_directory data/NoisySpeechSpectrograms
python3 generate_spectrograms.py --input_directory data/CleanSpeech --output_directory data/CleanSpeechSpectrograms