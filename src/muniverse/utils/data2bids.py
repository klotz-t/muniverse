import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
#from edfio import Edf, Bdf, EdfSignal, read_edf, read_bdf
from pyedflib.highlevel import read_edf, write_edf, make_header


class bids_dataset:

    def __init__(
        self, 
        datasetname="dataset_name", 
        root="./", 
        n_digits=2, 
        overwrite=False
    ):

        self.root = root + datasetname
        self.datasetname = datasetname
        self.dataset_sidecar = {
            "Name": datasetname,
            "BIDSVersion": self._get_bids_version(),
        }
        self.subjects_sidecar = self._set_participant_sidecar()
        self.subjects_data = pd.DataFrame(
            columns=["participant_id", "age", "sex", "handedness", "weight", "height"]
        )
        self.n_digits = n_digits
        self.overwrite = overwrite

    def write(self):
        """
        Export BIDS dataset

        """

        # make folder
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        # write participant.tsv
        name = self.root + "/" + "participants.tsv"
        if self.overwrite or not os.path.isfile(name):
            self.subjects_data.to_csv(name, sep="\t", index=False, header=True, na_rep="n/a")
        elif not self.overwrite and os.path.isfile(name):
            from_file = pd.read_table(name)
            if not from_file.equals(self.subjects_data):
                self.subjects_data.to_csv(name, sep="\t", index=False, header=True, na_rep="n/a")
        # write participant.json
        name = self.root + "/" + "participants.json"
        if self.overwrite or not os.path.isfile(name):
            with open(name, "w") as f:
                json.dump(self.subjects_sidecar, f)
        # write dataset.json
        name = self.root + "/" + "dataset_description.json"
        if self.overwrite or not os.path.isfile(name):
            with open(name, "w") as f:
                json.dump(self.dataset_sidecar, f)

        return ()

    def read(self):
        """
        Import data from BIDS dataset

        """

        # read participant.tsv
        name = self.root + "/" + "participants.tsv"
        if os.path.isfile(name):
            self.subjects_data = pd.read_table(name, on_bad_lines="warn")
        # read participant.json
        name = self.root + "/" + "participants.json"
        if os.path.isfile(name):
            with open(name, "r") as f:
                self.subjects_sidecar = json.load(f)
        # read dataset.json
        name = self.root + "/" + "dataset_description.json"
        if os.path.isfile(name):
            with open(name, "r") as f:
                self.dataset_sidecar = json.load(f)

        return ()

    def set_metadata(self, field_name, source):
        """
        Generic metadata update function.

        Parameters:
            field_name (str): name of the metadata attribute to update
            source (dict, DataFrame, or str): data or file path
        """
        current = getattr(self, field_name, None)
        if current is None:
            raise ValueError(f"No such field '{field_name}'")

        # Load from file if needed
        if isinstance(source, str):
            if source.endswith(".json"):
                with open(source) as f:
                    source = json.load(f)
            elif source.endswith(".tsv"):
                source = pd.read_csv(source, sep="\t")
            else:
                raise ValueError(f"Unsupported file type: {source}")

        # Update logic based on current type
        if isinstance(current, dict):
            if isinstance(source, dict):
                current.update(source)
            elif isinstance(source, pd.DataFrame):
                current.update(source.to_dict(orient="records")[0])  # assumes one row
            else:
                raise TypeError("Expected dict or DataFrame for dict field")
        elif isinstance(current, pd.DataFrame):
            if isinstance(source, dict):
                row = pd.DataFrame(data=source)
                current = pd.concat([current, row], ignore_index=True)
            elif isinstance(source, pd.DataFrame):
                current = pd.concat([current, source], ignore_index=True)
            else:
                raise TypeError("Expected dict or DataFrame for DataFrame field")
        else:
            raise TypeError(f"Unsupported target type for '{field_name}'")

        if field_name == "subjects_data":
            current.drop_duplicates(subset="participant_id", keep="last")
            current.sort_values("participant_id")

        # Update field
        setattr(self, field_name, current)

    def _get_label_from_filename(self, file, key):
        """
        Extract a label of some key in a BIDS filename

        Args: 
            file (str): BIDS filename
            key (str): key from which the label is extracted (e.g., "sub" or "task")

        Returns: 
            label (str): label corresponding to the requested BIDS key
        """    

        filename = Path(file).name
        label = re.search(fr"{key}-([^-_]+)", filename)

        return label
        


    def list_all_files(self, suffix, extension):
        """
        Summarize all files with a given extension that are part of a BIDS folder

        Args:
            suffix (str): Data type (e.g., emg)
            extension (str): File extension to be filtered (e.g. 'edf' for all *.edf files)

        Returns:
            df (DataFrame): Data frame listing all files in the given folder
        """

        root = Path(self.root)
        files = list(root.rglob(f"*{suffix}.{extension}"))
        filenames = [f.name for f in files]
        paths = [f.resolve() for f in files]
        columns = [
            "sub","ses","task","acq","run","recording","suffix","extension","file_path","file_name","dataset_name"
        ]
        df = pd.DataFrame(np.nan, index=range(len(filenames)), columns=columns)
        df = df.astype(
            {
                "sub": "string",
                "ses": "string",
                "task": "string",
                "acq": "string",
                "run": "string",
                "recording": "string",
                "suffix": "string",
                "extension": "string",
                "file_path": "string",
                "file_name": "string",
                "dataset_name": "string"
            }
        )

        for i in np.arange(len(filenames)):
            fname = str(paths[i])
            df.loc[i, "file_path"] = str(paths[i].parent)
            df.loc[i, "file_name"] = str(paths[i])
            df.loc[i, "sub"] = self._get_label_from_filename(fname, "sub")
            label = self._get_label_from_filename(fname, "ses")
            df.loc[i, "ses"] = label if label else pd.NA
            df.loc[i, "task"] = self._get_label_from_filename(fname, "task")
            label = self._get_label_from_filename(fname, "acq")
            df.loc[i, "acq"] = label if label else pd.NA
            label = self._get_label_from_filename(fname, "run")
            df.loc[i, "run"] = label if label else pd.NA
            label = self._get_label_from_filename(fname, "recording")
            df.loc[i, "recording"] = label if label else pd.NA
            df.loc[i, "suffix"] = suffix
            df.loc[i, "extension"] = extension
            df.loc[i, "dataset_name"] = self.datasetname

        return df

    def _set_participant_sidecar(self):
        """
        Return a template for initalizing the participant sidecar file

        """

        metadata = {
            "participant_id": {
                "Description": "Unique subject identifier"
            },
            "age": {
                "Description": "Age of the participant at time of testing",
                "Unit": "years",
            },
            "sex": {
                "Description": "Biological sex of the participant",
                "Levels": {"F": "female", "M": "male", "O": "other"},
            },
            "handedness": {
                "Description": "handedness of the participant as reported by the participant",
                "Levels": {"L": "left", "R": "right"},
            },
            "weight": {
                "Description": "Body weight of the participant", 
                "Unit": "kg"
            },
            "height": {
                "Description": "Body height of the participant", 
                "Unit": "m"
            },
        }

        return metadata
    
    def _get_bids_version(self):
        """
        Get the BIDS version of your dataset

        """

        bids_version = "1.11.1"

        return bids_version


class bids_emg_recording(bids_dataset):
    """
    Class for handling EMG recordings in BIDS format.

    This class implements the BIDS standard for EMG data, including support for root, subject
    or session-level inheritance of metadata files. By default, all metadata files are linked
    to their respective recording files. However, certain metadata files can be inherited
    at the session level to avoid duplication.

    Inheritance Rules:
    - By default, no metadata files are inherited (all are linked to _emg.edf)
    - emg.json, channels.tsv, electrodes.tsv and coordsystem.json can be inherited at dataset, subject or session level
    - Inherited files are stored at dataset, subject session level with names like:
      -> root/dataset/electrodes.tsv
      -> root/dataset/sub-01/sub-01_electrodes.tsv
      -> root/dataset/sub-01/ses-01/sub-01_ses-01_electrodes.tsv
    - Non-inherited files are stored with recording files like:
      root/dataset/sub-01/ses-01/emg/sub-01_ses-01_task-rest_run-01_electrodes.tsv
    """

    # Define valid metadata files that can be inherited and valid inheritance levels
    INHERITABLE_FILES = ["emg", "channels" ,"electrodes", "coordsystem"]
    INHERITABLE_LEVELS = ["dataset" , "task", "subject", "session"]
    # Define permissible raw data formats
    FILE_FORMATS = ["edf", "edf+", "bdf", "bdf+"]

    def __init__(
        self,
        subject_label="01",
        session_label=None,
        task_label="taskDescription",
        acq_label = None,
        run_label="01",
        recording_label=None,
        datatype="emg",
        root="./",
        datasetname="dataset_name",
        fileformat="edf",
        overwrite=False,
        inherited_metadata=None,
        inherited_level=None,
        dataset_config=None
    ):

        super().__init__(
            datasetname=datasetname,
            root=root,
            overwrite=overwrite,
        )

        if isinstance(dataset_config, bids_dataset):
            self.root = dataset_config.root
            self.datasetname = dataset_config.datasetname
            self.subjects_data = dataset_config.subjects_data

        # Check if the function arguments are valid
        self._validate_arguments(
            subject_label, session_label, task_label,  acq_label, run_label, recording_label, datatype
        )

        # Get the datapath
        if session_label is None:
            datapath = f"{self.root}/sub-{subject_label}/{datatype}/"
        else:
            datapath = f"{self.root}/sub-{subject_label}/ses-{session_label}/{datatype}/"

        # Store essential information for BIDS compatible folder structure in a dictonary
        self.datapath = datapath
        self.subject_label = subject_label
        self.session_label = session_label
        self.task_label = task_label
        self.acq_label = acq_label
        self.run_label = run_label
        self.recording_label = recording_label
        self.datatype = datatype
        #self.emg_data = Edf([EdfSignal(np.zeros(1), sampling_frequency=1)])
        self.emg_data = np.empty([1,1])
        if fileformat in self.FILE_FORMATS:
            self.fileformat = fileformat
        else:
            raise ValueError(f"Invalid fileformat: {fileformat}")

        # Initialize metadata
        self.channels = pd.DataFrame(columns=["name", "type", "unit"])
        self.electrodes = pd.DataFrame(
            columns=["name", "x", "y", "z", "coordinate_system"]
        )
        self.emg_sidecar = self._init_emg_sidecar()
        self.coord_sidecar = {
            "mycoordsystemname": {
                "EMGCoordinateSystem": "Other",
                "EMGCoordinateSystemDescription": "Free-form text description of the coordinate system", 
                "EMGCoordinateUnits": "mm"
            }
        }

        # Initialize empty inheritance dictionary
        self.inherited_metadata = {}
        self.inherited_levels = {}

        # Set inherited metadata if provided
        if inherited_metadata is not None:
            self._set_inherited_metadata(inherited_metadata, inherited_level)

    def _set_inherited_metadata(self, metadata_files, inherited_level):
        """
        Set which metadata files should be inherited at session level.

        Parameters:
        -----------
        metadata_files : list of str
            - List of metadata file names to inherit. Must be from INHERITABLE_FILES.
            Example: ['electrodes', 'coordsystem']
        inherited_level: list of str 
            - Level of the inherited metadata. Must be from INHERITABLE_LEVELS
            Examples: ['session', 'subject'] or ['dataset', 'dataset]

        Notes:
        ------
        - Only emg.sjon, channels.tsv, electrodes.tsv and coordsystem.json can be inherited
        - Inherited files are stored at dataset, subject of session level
        - Non-inherited files are stored with their respective recording files
        """
        # Validate input
        invalid_files = [f for f in metadata_files if f not in self.INHERITABLE_FILES]
        if invalid_files:
            raise ValueError(
                f"Invalid metadata files for inheritance: {invalid_files}. "
                f"Valid options are: {self.INHERITABLE_FILES}"
            )
        
        invalid_levels = [f for f in inherited_level if f not in self.INHERITABLE_LEVELS]
        if invalid_levels:
            raise ValueError(
                f"Invalid metadata files for inheritance: {invalid_levels}. "
                f"Valid options are: {self.INHERITABLE_LEVELS}"
            )
        
        if len(inherited_level) != len(metadata_files):
            raise ValueError(
                "The length of inherited_metadata and inherited_level must be identical."
            )


        # Set inheritance flags
        self.inherited_metadata = {f: True for f in metadata_files}
        self.inherited_levels = {f: inherited_level[i] for i, f in enumerate(metadata_files)}

    def _get_bids_filename(self, datatype, extension):
        """
        Get a BIDS compatible filename
        
        """

        # Non inherited metdadata files
        if not self.inherited_metadata.get(datatype, False) or extension in self.FILE_FORMATS: 
            fname = f"sub-{self.subject_label}_"
            folder = f"{self.root}/sub-{self.subject_label}/" 

            if self.session is not None:
                fname = fname + f"ses-{self.session_label}_"
                folder = folder + f"ses-{self.session_label}/"

            folder = folder + f"{self.datatype}/"    

            fname = fname + f"task-{self.task_label}_"

            if self.acq_label is not None:
                fname = fname + f"acq-{self.acq_label}_"
            
            if self.run_label is not None :
                fname = fname + f"run-{self.run_label}_"

            if self.recording_label is not None :
                fname = fname + f"recording-{self.recording_label}_"    

        # Inherited metdadata files
        else:
            level = self.inherited_levels[datatype]
            if level == "dataset":
                fname = ""
                folder = f"{self.root}/"
            elif level == "task":
                fname = self.task_label
                folder = f"{self.root}/"
            elif level == "subject":
                fname = f"sub-{self.subject_label}_"
                folder = f"{self.root}/sub-{self.subject_label}/"
            elif level == "session":
                fname = f"sub-{self.subject_label}_ses-{self.session_label}_"
                folder = f"{self.root}/sub-{self.subject_label}/ses-{self.session_label}/"
           
        if datatype is None:
            pass
        elif extension is None:
            fname = fname + f"{datatype}"
        else:
            fname = fname + f"{datatype}.{extension}" 

        name = folder + fname   

        return name
    
    def _find_inherited_file(self, ending):
        """
        Automatically find metadata files that are potentially inherited.
        

        """

        warnings.warn(
            f"File *_{ending} could not be found in the expected folder."
            "Trying to automatically search for inherited files."
        )

        searchpath = Path(self.datapath).parent.resolve()
        stop_folder = Path(self.root).resolve()

        files = list()
        
        while True:
            # search in current folder
            for file in searchpath.glob(f"*{ending}"):
                files.append(file.resolve())

            if len(files) > 0:
                return files   

            # stop if we reached the defined top-level folder
            if searchpath == stop_folder:
                break

            # stop if we reached filesystem root (safety)
            if searchpath.parent == searchpath:
                break

            # go one level up
            searchpath = searchpath.parent

        return None
    
    def _validate_arguments(
            self, 
            subject_label, 
            session_label,
            task_label,
            acq_label, 
            run_label, 
            recording_label,
            datatype):
        """
        Return error if the selected arguments are invalid
        
        """

        if type(subject_label) is not str:
            raise ValueError("subject_label must be of type string")
        
        if type(task_label) is not str:
            raise ValueError("task_label must be of type string")

        if session_label is not None:
            if type(session_label) is not str:
                raise ValueError("session_label must be of type string or None")
            
        if acq_label is not None:
            if type(acq_label) is not str:
                raise ValueError("acq_label must be of type string or None")    

        if run_label is not None:
            if type(run_label) is not str:
                raise ValueError("run_label must be of type string or None")
            
        if recording_label is not None:
            if type(recording_label) is not str:
                raise ValueError("recording_label must be of type string or None")    

        if datatype not in ["emg"]:
            raise ValueError("datatype must be emg")

    def _init_emg_sidecar(self):
        """
        Initalize the required EMG sidecar metadata

        """

        metadata = {
            "EMGPlacementScheme": "How electrode positions are determined (ChannelSpecific, Measured or Other)",
            "EMGPlacementSchemeDescription": "Details about EMG sensor placement",
            "EMGReference": "Description of the approach to signal referencing",
            "SamplingFrequency": 2048,
            "PowerLineFrequency": 50,
            "SoftwareFilters": "Object of temporal software filters applied, or n/a if the data is not available",
            "RecordingType": "Data continuity (continuous, epoched or discontinuous)",
        }

        return metadata

    def write(self):
        """
        Save dataset in BIDS format
        
        """
        super().write()

        # Generate an empty set of folders for your BIDS dataset
        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)

        # Write files
        filename = self._get_bids_filename("channels", "tsv")
        self.channels.to_csv(filename, sep="\t", index=False, header=True, na_rep="n/a")

        filename = self._get_bids_filename("emg","json")
        with open(filename, "w") as f:
            json.dump(self.emg_sidecar, f)

        filename = self._get_bids_filename("electrodes","tsv")
        self.electrodes.to_csv(filename, sep="\t", index=False, header=True, na_rep="n/a")

        filename = self._get_bids_filename("space-", None)
        for name, metadata in vars(self.coord_sidecar).items():
            filename2 = f"{filename}{name}_coordsystem.json"
            with open(filename2, "w") as f:
                json.dump(metadata, f)

        filename = self._get_bids_filename("emg", self.fileformat)
        #self.emg_data.write(filename)
        if self.emg_data.shape[0] != len(self.channels):
            channel_names = [f"Ch{i}" for i in range(self.emg_data.shape[0])]
        else:
            channel_names = self.channels["name"].values.tolist()
        signal_headers = make_header(
            channel_names, 
            sample_frequency=self.emg_sidecar["SamplingFrequency"]
        )
        write_edf(filename, self.emg_data, signal_headers)

    def read(self):
        """
        Import data from BIDS dataset
        
        """
        super().read()

        filename = self._get_bids_filename("channels", "tsv")
        if os.path.isfile(filename):
            self.channels = pd.read_table(filename, on_bad_lines="warn")
        else:
            filename = self._find_inherited_file("channels.tsv")[0]
            if os.path.isfile(filename):
                self.channels = pd.read_table(filename, on_bad_lines="warn")

        filename = self._get_bids_filename("emg","json")    
        if os.path.isfile(filename):
            with open(filename, "r") as f:
                self.emg_sidecar = json.load(f)
        else:
            filename = self._find_inherited_file("emg.json")[0]
            with open(filename, "r") as f:
                self.emg_sidecar = json.load(f)   

        filename = self._get_bids_filename("electrodes","tsv")
        if os.path.isfile(filename):
            self.electrodes = pd.read_table(filename, on_bad_lines="warn")
        else:
            filename = self._find_inherited_file("electrodes.tsv")[0]
            if os.path.isfile(filename):
                self.electrodes = pd.read_table(filename, on_bad_lines="warn")    

        serach_parts = [Path(self._get_bids_filename(None, None)).name, "coordsystem"]
        filename = [f for f in Path(self.datapath).iterdir()
                    if all(part in f.name for part in serach_parts)]
        if len(filename) == 0:
            filename = self._find_inherited_file("coordsystem.json")
        self.coord_sidecar = {}
        for i in range(len(filename)):  
            if os.path.isfile(filename[i]):
                coordname = re.search(fr"space-([^-_]+)", filename[i].name)
                if coordname is None:
                    coordname = "global"
                else:
                    coordname = coordname.group(1)
                with open(filename[i], "r") as f:
                    self.coord_sidecar.update({coordname: json.load(f)})

        filename = self._get_bids_filename("emg", self.fileformat)
        if os.path.isfile(filename):
            #self.emg_data = read_edf(filename)
            self.emg_data, _, _ = read_edf(filename)

    def set_metadata(self, field_name, source):

        super().set_metadata(field_name, source)

        if field_name == "channels":
            # Drop duplicates
            self.channels = self.channels.drop_duplicates(subset="name", keep="last")
        elif field_name == "electrodes":
            # Drop duplicates
            self.electrodes = self.electrodes.drop_duplicates(
                subset=["name", "coordinate_system"], keep="last"
            )
            # If the z coordinate exist make sure the ordering is correct
            if "z" in self.electrodes.columns:
                col = self.electrodes.pop("z")
                self.electrodes.insert(3, "z", col)

    def set_data(self, field_name, mydata, fsamp):
        """
        Add raw data and convert it into edf format

        Args:
            field_name (str): name of the field to be updated
            mydata (np.ndarry): data matrix (n_samples x n_channels)
            fsamp (float): Sampling frequency in Hz

        """

        # Add zeros to the signal such that the total length is in full seconds
        seconds = np.ceil(mydata.shape[0] / fsamp)
        signal = np.zeros([int(seconds * fsamp), mydata.shape[1]])
        signal[0 : mydata.shape[0], :] = mydata

        ## Initalize
        #edf = Edf([EdfSignal(signal[:, 0], sampling_frequency=fsamp)])
        #
        #for i in np.arange(1, signal.shape[1]):
        #    new_signal = EdfSignal(signal[:, i], sampling_frequency=fsamp)
        #    edf.append_signals(new_signal)

        # Set data
        setattr(self, field_name, signal)

        # Update Sampling Rate
        self.emg_sidecar["SamplingFrequency"] = fsamp

        return ()

    def read_data_frame(self, df, idx):
        self.subject_label = df.loc[idx, "sub"]
        label = df.loc[idx, "ses"]
        self.session_label = label if label else None
        self.task_label = df.loc[idx, "task"]
        label = df.loc[idx, "acq"]
        self.acq_label = label if label else None
        label = df.loc[idx, "run"]
        self.run_label = label if label else None
        label = df.loc[idx, "recording"]
        self.recording_label = label if label else None
        self.datatype = df.loc[idx, "suffix"]
        self.datapath = df.loc[idx, "file_path"] + "/"
        self.datasetname = df.loc[idx, "dataset_name"]

        self.read()
        
        return ()

class bids_neuromotion_recording(bids_emg_recording):
    """
    Class for handling neuromotion simulation data in BIDS format.
    Inherits from bids_emg_recording and adds support for additional simulation-specific files.
    """

    def __init__(
        self,
        subject_id=1,
        subject_desc="sim",
        task_label="isometric",
        datatype="emg",
        session=None,
        run=1,
        dataset_config=None,
        root="./",
        datasetname="dataset_name",
        fileformat="edf",
        overwrite=False,
        n_digits=2,
        inherited_metadata=None,
    ):

        # If no inherited_metadata is provided, use all inheritable files
        if inherited_metadata is None:
            inherited_metadata = self.INHERITABLE_FILES

        super().__init__(
            subject_id=subject_id,
            subject_desc=subject_desc,
            task_label=task_label,
            datatype=datatype,
            session=session,
            run=run,
            dataset_config=dataset_config,
            root=root,
            datasetname=datasetname,
            fileformat=fileformat,
            overwrite=overwrite,
            n_digits=n_digits,
            inherited_metadata=inherited_metadata,
        )


        # Initialize additional simulation-specific attributes
        self.spikes = pd.DataFrame(columns=["source_id", "spike_time"])
        self.motor_units = pd.DataFrame(
            columns=[
                "source_id",
                "recruitment_threshold",
                "depth",
                "innervation_zone",
                "fibre_density",
                "fibre_length",
                "conduction_velocity",
                "angle",
            ]
        )
        #self.internals = Edf([EdfSignal(np.zeros(1), sampling_frequency=1)])
        self.internals = np.empty([1,1])
        self.internals_sidecar = pd.DataFrame(
            columns=["name", "type", "units", "description"]
        )
        self.simulation_sidecar = {}

    def write(self):
        """Override write method to include simulated data"""
        # Call parent's write method to handle standard BIDS files
        super().write()

        # Write simulation-specific files
        filename = self._get_bids_filename("spikes","tsv")
        self.spikes.to_csv(filename, sep="\t", index=False, header=True, na_rep="n/a")
        filename = self._get_bids_filename("motorunits","tsv")
        self.motor_units.to_csv(filename, sep="\t", index=False, header=True, na_rep="n/a")
        filename = self._get_bids_filename("internals","tsv")
        self.internals_sidecar.to_csv(filename, sep="\t", index=False, header=True, na_rep="n/a")
        filename = self._get_bids_filename("simulation","json")
        with open(filename, "w") as f:
            json.dump(self.simulation_sidecar, f)
        filename = self._get_bids_filename("internals","edf")    
        #self.internals.write_edf(filename)

        channel_names = [f"{i}" for i in range(self.internals.shape[0])]
        signal_headers = make_header(
            channel_names, 
            sample_frequency=self.emg_sidecar["SamplingFrequency"]
        )
        write_edf(filename, self.internals, signal_headers)

    def read(self):
        """Override read method to include simulated data"""
        # Call parent's read method first
        super().read()

        # Read simulation-specific files
        filename = self._get_bids_filename("spikes","tsv")
        if os.path.isfile(filename):
            self.spikes = pd.read_table(filename, on_bad_lines="warn")
        filename = self._get_bids_filename("motorunits","tsv")    
        if os.path.isfile(filename):
            self.motor_units = pd.read_table(
                filename, on_bad_lines="warn"
            )
        filename = self._get_bids_filename("internals","tsv")    
        if os.path.isfile(filename):
            self.internals_sidecar = pd.read_table(
                filename, on_bad_lines="warn"
            )
        filename = self._get_bids_filename("simulation","json")    
        if os.path.isfile(filename):
            with open(filename, "r") as f:
                self.simulation_sidecar = json.load(f)
        filename = self._get_bids_filename("internals", self.fileformat)
        if os.path.isfile(filename):
            #self.internals = read_edf(filename)
            self.internals, _, _ = read_edf(filename)



class bids_decomp_derivatives(bids_emg_recording):

    def __init__(
        self,
        pipelinename="pipeline-name",
        format="standalone",
        rec_config=None,
        datasetname="dataset_name",
        datatype="emg",
        subject_id=1,
        subject_desc = "",
        task_label="isometric",
        run=1,
        session=None,
        desc_label="decomposed",
        root="./",
        fileformat="edf",
        overwrite=False,
        n_digits=2,
    ):

        # Check if the function arguments are valid
        self._validate_arguments(subject_id, session, run, datatype, n_digits)

        # Process name and session input
        subject_name = f"sub-{subject_desc}{self._id_to_label(subject_id)}"
        if session is None:
            datapath = f"{subject_name}/{datatype}/"
        else:
            session_name = f"ses-{self._id_to_label(session)}" 
            datapath = f"{subject_name}/{session_name}/{datatype}/"

        # Store essential information for BIDS compatible folder structure in a dictonary
        self.datapath = datapath
        self.subject_label = f"{subject_desc}{self._id_to_label(subject_id)}"
        self.task = "task-" + task_label
        self.session = session
        self.desc = "desc-" + desc_label
        self.overwrite = overwrite
        self.n_digits = n_digits
        self.run = run
        self.datatype = datatype
        self.fileformat = fileformat

        # Adopt labels from an emg recording in BIDS format
        if isinstance(rec_config, bids_emg_recording):
            self.root = rec_config.root
            self.datasetname = rec_config.datasetname
            self.n_digits = rec_config.n_digits
            self.subject_label = rec_config.subject_label
            self.task_label = rec_config.task_label
            self.run = rec_config.run
            self.datatype = rec_config.datatype
            self.session = rec_config.session

        # Make a BIDS compatible folder structure
        if format == "standalone":
            self.datasetname = f"{datasetname}-{pipelinename}"
            self.root = f"{root}{self.datasetname}/"

        else:
            self.datasetname = datasetname
            self.root = f"{root}{datasetname}/derivatives/{pipelinename}/"

        self.derivative_datapath = self.root + datapath
        self.pipelinename = pipelinename

        #self.source = Edf([EdfSignal(np.zeros(1), sampling_frequency=1)])
        self.source = np.empty([1, 1])
        self.spikes = pd.DataFrame(columns=["unit_id", "spike_time", "timestamp"])
        self.pipeline_sidecar = {"PipelineName": pipelinename}
        self.dataset_sidecar = {
            "Name": datasetname + "_" + pipelinename,
            "BIDSVersion": self._get_bids_version(),
            "GeneratedBy": pipelinename,
        }

    def write(self):
        """
        Save dataset in BIDS format

        """
        # Generate an empty set of folders for your BIDS dataset
        if not os.path.exists(self.derivative_datapath):
            os.makedirs(self.derivative_datapath)

        name = self.derivative_datapath + self.subject_name + "_"
        if self.session is not None:
            name = name + f"ses-{self._id_to_label(self.session)}_"
        name = name + self.task + "_"
        if self.run is not None:
            name = name + f"run-{self._id_to_label(self.run)}_"
        name = f"{name}_{self.desc}_"    

        # write *_predictedspikes.tsv
        self.spikes.to_csv(
            name + "events.tsv", sep="\t", index=False, header=True, na_rep="n/a"
        )
        # write *_pipeline.json
        fname = name + self.datatype + ".json"
        with open(fname, "w") as f:
            json.dump(self.pipeline_sidecar, f)
        # write *_predictedsources.edf file
        self.source.write(name + self.datatype + ".edf")
        # write dataset.json
        fname = self.root + "/dataset.json"
        if self.overwrite or not os.path.isfile(fname):
            with open(fname, "w") as f:
                json.dump(self.dataset_sidecar, f)

    def read(self):
        """
        Import data from BIDS dataset

        """
        # read *_predictedspikes.tsv
        name = self.derivative_datapath + self.subject_name + "_"
        if self.session > 0:
            name = name + f"ses-{self._id_to_label(self.session)}_"
        name = name + self.task + "_"
        if self.run > 0:
            name = name + f"run-{self._id_to_label(self.run)}_"
        name = f"{name}_{self.desc}_"         

        # read *_predictedspikes.tsv
        fname = name + "events.tsv"
        if os.path.isfile(fname):
            self.spikes = pd.read_table(fname, on_bad_lines="warn")
        # read *_pipeline.json
        fname = name + self.datatype + ".json"
        if os.path.isfile(fname):
            with open(fname, "r") as f:
                self.pipeline_sidecar = json.load(f)
        # read *.edf file
        fname = name + self.datatype + ".edf"
        if os.path.isfile(fname):
            #self.source = read_edf(fname)
            self.source, _, _ = read_edf(fname)
        # read dataset.json
        fname = self.root + "/" + "dataset.json"
        if os.path.isfile(fname):
            with open(fname, "r") as f:
                self.dataset_sidecar = json.load(f)

    def add_spikes(self, spikes):
        """
        Convert a dictionary of spike times to long-format TSV-style DataFrame.

        Parameters:
            spike_dict (dict): {source_id: list of spike times}

        """
        rows = []
        for unit_id, spike_times in spikes.items():
            for t in spike_times:
                rows.append({"source_id": unit_id, "spike_time": t})

        frames = [self.spikes, pd.DataFrame(rows)]
        self.spikes = pd.concat(frames, ignore_index=True)
        self.spikes = self.spikes.drop_duplicates(subset=["source_id", "spike_time"])


#def edf_to_numpy(edf_data, idx):
#    """
#    Output data of selcetd channels as numpy array
#
#    Args:
#        edf_data (edf): Time series data in edf format
#        idx (ndarray): Indices of the channels to be stored
#
#    Returns:
#        np_data (np.ndarray): Time series data
#    """
#
#    np_data = np.zeros((edf_data.signals[idx[0]].data.shape[0], len(idx)))
#    for i in np.arange(len(idx)):
#        np_data[:, i] = edf_data.signals[idx[i]].data
#
#    return np_data
