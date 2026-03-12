import json
import os
import re
import warnings
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
#from edfio import Edf, Bdf, EdfSignal, read_edf, read_bdf
from pyedflib.highlevel import read_edf, write_edf, make_signal_headers


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
        label = label.group(1) if label else None

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
    
    def is_valid(self):
        """
        Returns True if your BIDS dataset is valid (i.e., has no errors). 
        
        """

        errors, _ = run_bids_validator(
            self.root + "/", 
            print_errors=False, 
            print_warnings=False
        )

        if len(errors) == 0:
            return True
        else:
            return False
    
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
    INHERITABLE_FILES = ["emg", "channels" ,"electrodes", "space"]
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
        fsamp=2048,
        plfreq=50,
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
        self.emg_sidecar = self._init_emg_sidecar(fsamp, plfreq)
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

            if self.session_label is not None:
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

    def _init_emg_sidecar(self, fsamp, plfreq):
        """
        Initalize the required EMG sidecar metadata

        """

        metadata = {
            "EMGPlacementScheme": "How electrode positions are determined (ChannelSpecific, Measured or Other)",
            "EMGPlacementSchemeDescription": "Details about EMG sensor placement",
            "EMGReference": "Description of the approach to signal referencing",
            "SamplingFrequency": fsamp,
            "PowerLineFrequency": plfreq,
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

        filename = self._get_bids_filename("space", None)
        for name, metadata in self.coord_sidecar.items():
            filename2 = f"{filename}-{name}_coordsystem.json"
            with open(filename2, "w") as f:
                json.dump(metadata, f)

        filename = self._get_bids_filename("emg", self.fileformat)
        #self.emg_data.write(filename)
        if self.emg_data.shape[0] != len(self.channels):
            channel_names = [f"Ch{i}" for i in range(self.emg_data.shape[0])]
        else:
            channel_names = self.channels["name"].values.tolist()
        signal_headers = make_signal_headers(
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
            mydata (np.ndarry): data matrix (n_channels x n_samples)
            fsamp (float): Sampling frequency in Hz

        """

        # Add zeros to the signal such that the total length is in full seconds
        seconds = np.ceil(mydata.shape[1] / fsamp)
        signal = np.zeros([mydata.shape[0], int(seconds * fsamp)])
        signal[:, 0 : mydata.shape[1]] = mydata

        setattr(self, field_name, signal)

        if field_name == "emg_data":
            self.emg_sidecar["SamplingFrequency"] = fsamp


    def read_data_frame(self, df, idx):
        self.subject_label = df.loc[idx, "sub"]
        label = df.loc[idx, "ses"]
        self.session_label = label if type(label) is str else None
        self.task_label = df.loc[idx, "task"]
        label = df.loc[idx, "acq"]
        self.acq_label = label if type(label) is str else None
        label = df.loc[idx, "run"]
        self.run_label = label if type(label) is str else None
        label = df.loc[idx, "recording"]
        self.recording_label = label if type(label) is str else None
        self.datatype = df.loc[idx, "suffix"]
        self.datapath = df.loc[idx, "file_path"] + "/"
        self.datasetname = df.loc[idx, "dataset_name"]

        self.read()


class bids_neuromotion_recording(bids_emg_recording):
    """
    Class for handling neuromotion simulation data in BIDS format.
    Inherits from bids_emg_recording and adds support for additional simulation-specific files.
    """

    BIDSIGNORE = [
        "*_spikes.tsv",
        "*_motorunits.tsv",
        "*_internals.tsv",
        "*_internals.edf",
        "*_simulation.json"
    ]

    def __init__(
        self,
        subject_label="sim01",
        session_label=None,
        task_label="isometric",
        acq_label=None,
        run_label="01",
        recording_label=None,
        datatype="emg",
        dataset_config=None,
        root="./",
        datasetname="dataset_name",
        fileformat="edf",
        fsamp=2048,
        plfreq=50,
        overwrite=False,
        inherited_metadata=None,
    ):

        # If no inherited_metadata is provided, use all inheritable files
        if inherited_metadata is None:
            inherited_metadata = self.INHERITABLE_FILES

        super().__init__(
            subject_label=subject_label,
            session_label=session_label,
            task_label=task_label,
            acq_label=acq_label,
            run_label=run_label,
            recording_label=recording_label,
            datatype=datatype,
            dataset_config=dataset_config,
            root=root,
            datasetname=datasetname,
            fsamp=fsamp,
            plfreq="n/a",
            fileformat=fileformat,
            overwrite=overwrite,
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
        signal_headers = make_signal_headers(
            channel_names, 
            sample_frequency=self.emg_sidecar["SamplingFrequency"]
        )
        write_edf(filename, self.internals, signal_headers)

        # write .bidsignore
        fname = Path(self.root) / ".bidsignore"
        fname.write_text("\n".join(self.BIDSIGNORE) + "\n")

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

    BIDSIGNORE = [
        "*_sources.edf",
        "*_sources.bdf",
        "*_sources.json",
        "descriptions.tsv"
    ]

    def __init__(
        self,
        pipelinename="pipeline-name",
        format="standalone",
        rec_config=None,
        datasetname="dataset_name",
        datatype="emg",
        subject_label="01",
        session_label=None,
        task_label="isometric",
        acq_label=None,
        run_label="01",
        recording_label=None,
        desc_label="decomposed",
        fsamp = 2048,
        root="./",
        fileformat="edf",
        overwrite=False,
    ):

        # Check if the function arguments are valid
        self._validate_arguments(
            subject_label, session_label, task_label, acq_label, run_label, recording_label, datatype
        )

        # Process name and session input
        if session_label is None:
            datapath = f"sub-{subject_label}/{datatype}/"
        else:
            datapath = f"sub-{subject_label}/ses-{session_label}/{datatype}/"

        # Store essential information for BIDS compatible folder structure in a dictonary
        self.datapath = datapath
        self.subject_label = subject_label
        self.session_label = session_label
        self.task_label = task_label
        self.acq_label = acq_label
        self.run_label = run_label
        self.recording_label = recording_label
        self.desc_label = desc_label
        self.overwrite = overwrite
        self.datatype = datatype
        self.fileformat = fileformat

        # Adopt labels from an emg recording in BIDS format
        if isinstance(rec_config, bids_emg_recording):
            self.root = rec_config.root
            self.datasetname = rec_config.datasetname
            self.subject_label = rec_config.subject_label
            self.session_label = rec_config.session_label
            self.task_label = rec_config.task_label
            self.acq_label = rec_config.acq_label
            self.run_label = rec_config.run_label
            self.recording_label = rec_config.recording_label
            self.datatype = rec_config.datatype

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
        self.events = pd.DataFrame(columns=["onset", "duration", "sample", "unit_id", "event_description"])
        self.events_sidecar = self._init_event_sidecar()
        self.descriptions = pd.DataFrame(columns=["desc_id", "description"])
        self.descriptions.loc[0, "desc_id"] = "decomposed"
        self.descriptions.loc[0, "description"] = "estimated spiking motor unit activity"
        self.source_sidecar = {
            "SamplingFrequency": fsamp,
            "NumberOfSources": 1
        }
        self.dataset_sidecar = {
            "Name": datasetname + "_" + pipelinename,
            "BIDSVersion": self._get_bids_version(),
            "GeneratedBy": [{
                "Name": pipelinename
            }]
        }

    def _init_event_sidecar(self):
        """
        Set up a template for the event sidecar json file
        
        """    

        metadata = {
            "onset": {
                "Description": "Onset time of the event in seconds from recording start.",
                "Unit": "s"
            },
            "duration": {
                "Description": "Duration of the event in seconds. A value of zero means that the event is a dirac pulse",
                "Unit": "s"
            },
            "sample": {
                "Description": "Sample index of the event onset (zero-indexing)",
                "Unit": "samples"
            },
            "unit_id": {
                "Description": "Unique identifier (integer value) of the motor unit corresponding to the detected spike.",
                "Unit": "Integer-based ID"
            },
            "event_description": {
                "Description": "Free text event description."
            }
        }

        return metadata

    def write(self):
        """
        Save dataset in BIDS format

        """
        # Generate an empty set of folders for your BIDS dataset
        if not os.path.exists(self.derivative_datapath):
            os.makedirs(self.derivative_datapath)

        name = self.derivative_datapath + f"sub-{self.subject_label}_"
        if self.session_label is not None:
            name = name + f"ses-{self.session_label}_"
        name = name + f"task-{self.task_label}_"
        if self.acq_label is not None:
            name = name + f"acq-{self.acq_label}_"
        if self.run_label is not None:
            name = name + f"run-{self.run_label}_"
        if self.recording_label is not None:
            name = name + f"recording-{self.recording_label}_"    
        name = f"{name}desc-{self.desc_label}_"    

        # write *_predictedspikes.tsv
        self.events.to_csv(
            name + "events.tsv", sep="\t", index=False, header=True, na_rep="n/a"
        )
        # write *_pipeline.json
        fname = name + "sources.json"
        with open(fname, "w") as f:
            json.dump(self.source_sidecar, f)
        # write *_desc-decomposed_emg.edf file
        filename = f"{name}sources.{self.fileformat}"
        #self.emg_data.write(filename)
        channel_names = [f"Unit_{i}" for i in range(self.source.shape[0])]
        signal_headers = make_signal_headers(
            channel_names, 
            sample_frequency=self.source_sidecar["SamplingFrequency"]
        )
        write_edf(filename, self.source, signal_headers)
        #self.source.write(name + self.datatype + self.fileformat)
        # write dataset.json
        fname = self.root + "dataset_description.json"
        if self.overwrite or not os.path.isfile(fname):
            with open(fname, "w") as f:
                json.dump(self.dataset_sidecar, f)

        # write events.json
        fname = self.root + "events.json"
        if self.overwrite or not os.path.isfile(fname):
            with open(fname, "w") as f:
                json.dump(self.events_sidecar, f)      

        # write descriptions.tsv
        fname = self.root + "descriptions.tsv" 
        if self.overwrite or not os.path.isfile(fname):
            self.descriptions.to_csv(
                fname, sep="\t", index=False, header=True, na_rep="n/a"
            )

        # write .bidsignore
        fname = Path(self.root) / ".bidsignore"
        fname.write_text("\n".join(self.BIDSIGNORE) + "\n")
                   

    def read(self):
        """
        Import data from BIDS dataset

        """
        
        #
        name = self.derivative_datapath + f"sub-{self.subject_label}_"
        if self.session_label is not None:
            name = name + f"ses-{self.session_label}_"
        name = name + f"task-{self.task_label}_"
        if self.acq_label is not None:
            name = name + f"acq-{self.acq_label}_"
        if self.run_label is not None:
            name = name + f"run-{self.run_label}_"
        if self.recording_label is not None:
            name = name + f"recording-{self.recording_label}_"    
        name = f"{name}desc-{self.desc_label}_"        

        # read *_predictedspikes.tsv
        fname = name + "events.tsv"
        if os.path.isfile(fname):
            self.events = pd.read_table(fname, on_bad_lines="warn")
        # read *_pipeline.json
        fname = name + "sources.json"
        if os.path.isfile(fname):
            with open(fname, "r") as f:
                self.source_sidecar = json.load(f)
        # read *.edf file
        fname = name + "sources" + self.fileformat
        if os.path.isfile(fname):
            #self.source = read_edf(fname)
            self.source, _, _ = read_edf(fname)
        # read dataset.json
        fname = self.root + "dataset_description.json"
        if os.path.isfile(fname):
            with open(fname, "r") as f:
                self.dataset_sidecar = json.load(f)
        # write events.json
        fname = self.root + "events.json"
        if os.path.isfile(fname):
            with open(fname, "r") as f:
                self.events_sidecar = json.load(f)          
        # read descriptions.tsv
        fname = self.root + "descriptions.tsv" 
        if os.path.isfile(fname):
            self.descriptions = pd.read_table(fname, on_bad_lines="warn")        

    def add_spikes(self, spikes, fsamp):
        """
        Convert a dictionary of spike times to long-format TSV-style DataFrame.

        Parameters:
            spike_dict (dict): {source_id: list of spike timings (in samples)}
            fsamp(float): Sampling frequency in Hz

        """
        rows = []
        for unit_id, spike_times in spikes.items():
            for t in spike_times:
                rows.append({
                    "onset": t/fsamp,
                    "duration": 0,
                    "sample": t,
                    "unit_id": unit_id, 
                    "event_description": "Motor-unit-spike"
                })

        frames = [self.events, pd.DataFrame(rows)]
        frames = [f for f in frames if not f.empty]
        self.events = pd.concat(frames, ignore_index=True)
        self.events = self.events.drop_duplicates(subset=["onset", "unit_id", "sample"])
        self.events = self.events.sort_values(by=["onset"])

    def set_data(self, field_name, mydata, fsamp):

        super().set_data(field_name, mydata, fsamp)

        if field_name == "source":
            self.source_sidecar["SamplingFrequency"] = fsamp
            self.source_sidecar["NumberOfSources"] = mydata.shape[0]

def run_bids_validator(
        path,
        ignored_codes = [],
        ignored_fields = [],
        ignored_files = [],
        print_errors=True, 
        print_warnings=True):
    """
    API to the official BIDS validator.

    Args:
        path (str): Absolute of relative path to your BIDS dataset 
        ignored_codes (list of str): Ignored error codes (e.g. ["SIDECAR_KEY_RECOMMENDED"])
        ignored_fileds (list of str): Errors corresponding to that field are ignored (e.g. ["DeviceSerialNumber"])
        ignored_files (list of str): Ignored errors in these files (e.g. ["/dataset_description.json"])
        print_errors (bool): Descides if errors should be printed 
        print_warnings (bool): Descides if warnings should be printed 

    Returns:
        errors (list): List of detected errors
        warnings (list): List of detected warnings    
    
    """

    result = subprocess.run(
        ["bids-validator-deno", "--format", "json", path],
        capture_output=True,
        text=True
    )

    validation = json.loads(result.stdout)

    messages = validation["issues"]["issues"]
    messages = [f for f in messages if f["code"] not in ignored_codes]
    #messages = [f for f in messages if f["subCode"] not in ignored_fields]
    messages = [f for f in messages if (not "subCode" in f or f["subCode"] not in ignored_fields)]
    messages = [f for f in messages if f["location"] not in ignored_files]
    errors = [f for f in messages if f["severity"] == "error"]
    warnings = [f for f in messages if f["severity"] == "warning"]

    if print_errors:
        print(f"Number of detected errors: {len(errors)}")
        print(json.dumps(errors, indent=4))
    if print_warnings:    
        print(f"Number of detected warnings: {len(warnings)}")
        print(json.dumps(warnings, indent=4))

    return errors, warnings

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
