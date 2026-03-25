import json
import os
import re
import warnings
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
#from edfio import Edf, Bdf, EdfSignal, read_edf, read_bdf
from pyedflib.highlevel import read_edf, write_edf, make_signal_headers


class bids_dataset:

    BIDSIGNORE = []

    def __init__(
        self, 
        datasetname="dataset_name", 
        root="./", 
        overwrite=False
    ):

        self.root = str(Path(root) / datasetname) + "/"
        self.datasetname = datasetname
        self.dataset_sidecar = {
            "Name": datasetname,
            "BIDSVersion": self._get_bids_version(),
            "DatasetType": "raw",
            "License": "The license for the dataset.",
            "Authors": ["Author 1", "Author 2", "..."]
        }
        self.readme = """# Some BIDS Dataset

        README is a required text file and should describe the dataset in more detail.

        The README file should be structured such that the dataset content can be easily understood.

        """ 
        self.subjects_sidecar = self._set_participant_sidecar()
        self.subjects_data = pd.DataFrame(
            columns=["participant_id", "age", "sex", "handedness", "weight", "height"]
        )
        self.overwrite = overwrite

    def write(self):
        """
        Export BIDS dataset

        """

        # make folder
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        # write participant.tsv
        name = f"{self.root}participants.tsv"
        if self.overwrite or not os.path.isfile(name):
            self.subjects_data.to_csv(name, sep="\t", index=False, header=True, na_rep="n/a")
        elif not self.overwrite and os.path.isfile(name):
            from_file = pd.read_table(name)
            if not from_file.equals(self.subjects_data):
                self.subjects_data.to_csv(name, sep="\t", index=False, header=True, na_rep="n/a")
        # write participant.json
        name = f"{self.root}participants.json"
        if self.overwrite or not os.path.isfile(name):
            with open(name, "w") as f:
                json.dump(self.subjects_sidecar, f, indent=4)
        # write dataset.json
        name = f"{self.root}dataset_description.json"
        if self.overwrite or not os.path.isfile(name):
            with open(name, "w") as f:
                json.dump(self.dataset_sidecar, f, indent=4)
        # write README.md
        name = f"{self.root}README.md"
        if self.overwrite or not os.path.isfile(name):
            with open(name, "w", encoding="utf-8") as f:
                f.write(self.readme)    

        # write .bidsignore
        if len(self.BIDSIGNORE) > 0:
            fname = Path(self.root) / ".bidsignore"
            if self.overwrite or not fname.exists():
                fname.write_text("\n".join(self.BIDSIGNORE) + "\n")         

        return ()

    def read(self):
        """
        Import data from BIDS dataset

        """

        # read participant.tsv
        name = f"{self.root}participants.tsv"
        if os.path.isfile(name):
            self.subjects_data = pd.read_table(name, on_bad_lines="warn")
        # read participant.json
        name = f"{self.root}participants.json"
        if os.path.isfile(name):
            with open(name, "r") as f:
                self.subjects_sidecar = json.load(f)
        # read dataset.json
        name = f"{self.root}dataset_description.json"
        if os.path.isfile(name):
            with open(name, "r") as f:
                self.dataset_sidecar = json.load(f)
        # read README.md
        name = f"{self.root}README.md"
        if os.path.isfile(name):
            with open(name, "r", encoding="utf-8") as f:
                self.readme = f.read()    
        # read .bidsignore   
        fname = Path(self.root) / ".bidsignore"
        if fname.exists():    
            self.BIDSIGNORE = fname.read_text(encoding="utf-8").splitlines()     

        return ()

    def set_metadata(self, field_name, source, overwrite=False):
        """
        Generic metadata update function.

        Parameters:
            field_name (str): name of the metadata attribute to update
            source (dict, DataFrame, or str): data or file path
            overwrite (bool): If False, the field is updated, otherwise overwritten by source
        """

        if field_name == "readme":
            raise ValueError(f"Property {field_name} is not supported by this function.")

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
            if overwrite:
                current = {}
            if isinstance(source, dict):
                current.update(source)
            elif isinstance(source, pd.DataFrame):
                current.update(source.to_dict(orient="records")[0])  # assumes one row
            else:
                raise TypeError("Expected dict or DataFrame for dict field")
        elif isinstance(current, pd.DataFrame):
            if overwrite:
                frames = [source]
            if isinstance(source, dict):
                row = pd.DataFrame(data=source)
                frames = [current, row]
            elif isinstance(source, pd.DataFrame):
                frames = [current, source]
            else:
                raise TypeError("Expected dict or DataFrame for DataFrame field")
            frames = [d for d in frames if not d.empty]
            current = pd.concat(frames, ignore_index=True)
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
            key (str): key from which the label is extracted (e.g., "sub", "task" or "run")

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
        
    def validate(
            self, 
            ignored_codes = [],
            ignored_fields = [],
            ignored_files = [],
            print_errors=True, 
            print_warnings=True
        ):
        """
        API to run the BIDS validator

        Args:
            ignored_codes (list of str): List of ignored error codes
            ignored_fields (list of str): List of ignored metadata fields
            ignored_files (list of str): List of ignored files
            print_errors (bool): If True print all errors
            print_errors (warnings): If True print all warnings
        Returns:
            err (list of objects): List of all errors
            war (list of objects): List of all warnings
            valid (bool): True if there are no errors
        
        """

        err, war, valid = run_bids_validator(
            f"{self.root}/",
            ignored_codes=ignored_codes,
            ignored_fields=ignored_fields,
            ignored_files=ignored_files,
            print_errors=print_errors,
            print_warnings=print_warnings
        )

        return err, war, valid   
    
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
    INHERITABLE_FILES = [
        "electrodes.tsv", "electrodes.json", 
        "coordsystem.json", "events.tsv", "events.json"
    ]
    INHERITABLE_LEVELS = ["dataset" , "task", "subject", "session"]
    # Define permissible raw data formats
    FILE_FORMATS = ["edf", "edf+", "bdf", "bdf+"]
    # Predefined fields in tabular columns
    CHANNELS_FIELDS = [
        "name", "type", "units", "description", "sampling_frequency",
        "signal_electrode", "reference", "group", "target_muscle",
        "placement_scheme", "placement_description", "interelectrode_distance", 
        "low_cutoff", "high_cutoff", "notch", "status", "status_description"
    ]
    ELECTRODES_FIELDS = [
        "name", "x", "y", "z", "coordinate_system", "type",
        "material", "impedance", "group"
    ]
    EVENTS_FIELDS = ["onset", "duration"]

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
        datasetname="datasetName",
        fileformat="edf",
        fsamp=2048,
        plfreq=50,
        overwrite=False,
        inherited_metadata=None,
        inherited_level=None,
        parent_dataset=None
    ):

        super().__init__(
            datasetname=datasetname,
            root=root,
            overwrite=overwrite,
        )

        if isinstance(parent_dataset, bids_dataset):
            self.root = parent_dataset.root
            self.datasetname = parent_dataset.datasetname
            self.subjects_data = parent_dataset.subjects_data

        # Check if the function arguments are valid
        self._validate_arguments(
            subject_label, session_label, task_label,  acq_label, run_label, recording_label, datatype
        )

        # Get the datapath
        if session_label is None:
            datapath = f"{self.root}sub-{subject_label}/{datatype}/"
        else:
            datapath = f"{self.root}sub-{subject_label}/ses-{session_label}/{datatype}/"

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
        self.fsamp = fsamp
        self.emg_data = np.empty((0,))
        if fileformat in self.FILE_FORMATS:
            self.fileformat = fileformat
        else:
            raise ValueError(f"Invalid fileformat: {fileformat}")

        # Initialize metadata
        self.channels = pd.DataFrame(columns=["name", "type", "units"])
        self.channels_sidecar = {}
        self.electrodes = pd.DataFrame(
            columns=["name", "x", "y", "z", "coordinate_system"]
        )
        self.electrodes_sidecar = {}
        self.emg_sidecar = self._init_emg_sidecar(fsamp, plfreq)
        self.coord_sidecar = {
            "templateCoordSystemName": {
                "EMGCoordinateSystem": "Other",
                "EMGCoordinateSystemDescription": "Free-form text description of the coordinate system", 
                "EMGCoordinateUnits": "mm"
            }
        }
        self.events = pd.DataFrame(columns=["onset", "duration"])
        self.events_sidecar = {}

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
        self.inherited_metadata = {
            ("space" if f == "coordsystem.json" else f): True
            for f in metadata_files
        }
        #self.inherited_metadata = {f: True for f in metadata_files}
        #self.inherited_levels = {f: inherited_level[i] for i, f in enumerate(metadata_files)}
        self.inherited_levels = {
            ("space" if f == "coordsystem.json" else f): inherited_level[i]
            for i, f in enumerate(metadata_files)
        }

    def _get_bids_filename(self, extension):
        """
        Get a BIDS compatible filename
        
        """

        # Non inherited metdadata files
        if not self.inherited_metadata.get(extension, False): 
            fname = f"sub-{self.subject_label}_"
            folder = f"{self.root}sub-{self.subject_label}/" 

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
            level = self.inherited_levels[extension]
            if level == "dataset":
                fname = ""
                folder = self.root
            elif level == "task":
                fname = self.task_label
                folder = self.root
            elif level == "subject":
                fname = f"sub-{self.subject_label}_"
                folder = f"{self.root}sub-{self.subject_label}/{self.datatype}/"
            elif level == "session":
                fname = f"sub-{self.subject_label}_ses-{self.session_label}_"
                folder = f"{self.root}sub-{self.subject_label}/ses-{self.session_label}/{self.datatype}"

        if extension is None:
            name = folder + fname
        else:
            name = folder + fname + extension

        return name
    
    def _find_inherited_file(self, ending):
        """
        Automatically find metadata files that are potentially inherited.
        

        """

        warnings.warn(
            f"File *_{ending} could not be found in the expected folder."
            "Trying to automatically search for inherited files."
        )

        searchpath = Path(self.datapath).resolve()
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
            "RecordingDuration": 0,
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
        filename = self._get_bids_filename("channels.tsv")
        if len(self.channels) > 0:
            self.channels.to_csv(filename, sep="\t", index=False, header=True, na_rep="n/a")
            if self.channels_sidecar:   
                filename = self._get_bids_filename("channels.json") 
                with open(filename, "w") as f:
                    json.dump(self.channels_sidecar, f, indent=4)

        filename = self._get_bids_filename("emg.json")
        with open(filename, "w") as f:
            json.dump(self.emg_sidecar, f, indent=4)
        
        if len(self.electrodes) > 0:
            filename = self._get_bids_filename("electrodes.tsv")
            self.electrodes.to_csv(filename, sep="\t", index=False, header=True, na_rep="n/a")
            # Coordinate systems metadata is only needed if the electrodes file exists
            filename = self._get_bids_filename("space")
            for name, metadata in self.coord_sidecar.items():
                filename2 = f"{filename}-{name}_coordsystem.json"
                with open(filename2, "w") as f:
                    json.dump(metadata, f, indent=4)
            # Electrodes sidecar is only needed if non pre-defined fields are used        
            if self.electrodes_sidecar:   
                filename = self._get_bids_filename("electrodes.json") 
                with open(filename, "w") as f:
                    json.dump(self.electrodes_sidecar, f, indent=4)        

        if self.emg_data.size > 0:
            filename = self._get_bids_filename(f"emg.{self.fileformat}")
            #self.emg_data.write(filename)
            if self.emg_data.shape[0] != len(self.channels):
                channel_names = [f"Ch{i}" for i in range(self.emg_data.shape[0])]
            else:
                channel_names = self.channels["name"].values.tolist()
            signal_headers = make_signal_headers(
                channel_names, 
                sample_frequency=self.fsamp
            )
            write_edf(filename, self.emg_data, signal_headers)

        # Write events.tsv and sidecar
        if len(self.events) > 0:
            filename = self._get_bids_filename("events.tsv")
            self.events.to_csv(filename, sep="\t", index=False, header=True, na_rep="n/a")

            name = self._get_bids_filename("events.json")
            if ((self.overwrite or not os.path.isfile(name)) and self.events_sidecar):
                with open(name, "w") as f:
                    json.dump(self.events_sidecar, f, indent=4)


    def read(self):
        """
        Import data from BIDS dataset
        
        """
        super().read()

        filename = self._get_bids_filename("channels.tsv")
        if os.path.isfile(filename):
            self.channels = pd.read_table(filename, on_bad_lines="warn")

        filename = self._get_bids_filename("channels.json")
        if os.path.isfile(filename):
            with open(filename, "r") as f:
                self.channels_sidecar = json.load(f)   

        filename = self._get_bids_filename("emg.json")    
        if os.path.isfile(filename):
            with open(filename, "r") as f:
                self.emg_sidecar = json.load(f)

        filename = self._get_bids_filename("electrodes.tsv")
        if os.path.isfile(filename):
            self.electrodes = pd.read_table(filename, on_bad_lines="warn")
        else:
            filename = self._find_inherited_file("electrodes.tsv")
            filename = filename[0] if filename else str()
            if os.path.isfile(filename):
                self.electrodes = pd.read_table(filename, on_bad_lines="warn")    

        filename = self._get_bids_filename("electrodes.json")
        if os.path.isfile(filename):
            with open(filename, "r") as f:
                self.electrodes_sidecar = json.load(f)
        else:
            filename = self._find_inherited_file("electrodes.json")
            filename = filename[0] if filename else str()
            if os.path.isfile(filename):
                with open(filename, "r") as f:
                    self.electrodes_sidecar = json.load(f)        

        serach_parts = [Path(self._get_bids_filename(None)).name, "coordsystem"]
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

        filename = self._get_bids_filename(f"emg.{self.fileformat}")
        if os.path.isfile(filename):
            #self.emg_data = read_edf(filename)
            self.emg_data, _ , _ = read_edf(filename)

        # Read events .tsv
        filename = self._get_bids_filename("events.tsv")
        if os.path.isfile(filename):
            self.events = pd.read_table(filename, on_bad_lines="warn")
        else:
            filename = self._find_inherited_file("events.tsv")
            filename = filename[0] if filename else str()
            if os.path.isfile(filename):
                self.events = pd.read_table(filename, on_bad_lines="warn") 

        filename = self._get_bids_filename("events.json")           
        if os.path.isfile(filename):
            with open(filename, "r") as f:
                self.events_sidecar = json.load(f)   
        else:
            filename = self._find_inherited_file("events.json")
            filename = filename[0] if filename else str()
            if os.path.isfile(filename):
                with open(filename, "r") as f:
                    self.events_sidecar = json.load(f)                  

    def set_metadata(self, field_name, source, overwrite=False):

        super().set_metadata(field_name, source, overwrite)

        if field_name == "channels":
            # Drop duplicates
            self.channels = self.channels.drop_duplicates(subset="name", keep="last")
            for col in self.channels.columns:
                if (col not in self.CHANNELS_FIELDS and col not in self.channels_sidecar):
                    warnings.warn(
                        f"Field {col} needs to be defined in channels_sidecar."
                    )
        elif field_name == "electrodes":
            # Drop duplicates
            self.electrodes = self.electrodes.drop_duplicates(
                subset=["name", "coordinate_system"], keep="last"
            )
            # If the z coordinate exist make sure the ordering is correct
            if "z" in self.electrodes.columns:
                col = self.electrodes.pop("z")
                self.electrodes.insert(3, "z", col)
            for col in self.electrodes.columns:
                if (col not in self.ELECTRODES_FIELDS and col not in self.electrodes_sidecar):
                    warnings.warn(
                        f"Field {col} needs to be defined in electrodes_sidecar."
                    )    
        elif field_name == "events":  
            # Drop duplicates
            self.events = self.events.drop_duplicates(keep="last")
            for col in self.events.columns:
                if (col not in self.EVENTS_FIELDS and col not in self.events_sidecar):
                    warnings.warn(
                        f"Field {col} needs to be defined in events_sidecar."
                    )
                  

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
            self.emg_sidecar["RecordingDuration"] = seconds
            self.fsamp = fsamp


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
        parent_dataset=None,
        root="./",
        datasetname="dataset_name",
        fileformat="edf",
        fsamp=2048,
        plfreq="n/a",
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
            parent_dataset=parent_dataset,
            root=root,
            datasetname=datasetname,
            fsamp=fsamp,
            plfreq=plfreq,
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
        self.internals = np.empty((0,))
        self.internals_sidecar = pd.DataFrame(
            columns=["name", "type", "units", "description"]
        )
        self.simulation_sidecar = {}

    def write(self):
        """Override write method to include simulated data"""
        # Call parent's write method to handle standard BIDS files
        super().write()

        # Write simulation-specific files
        filename = self._get_bids_filename("spikes.tsv")
        self.spikes.to_csv(filename, sep="\t", index=False, header=True, na_rep="n/a")
        filename = self._get_bids_filename("motorunits.tsv")
        self.motor_units.to_csv(filename, sep="\t", index=False, header=True, na_rep="n/a")
        filename = self._get_bids_filename("internals.tsv")
        self.internals_sidecar.to_csv(filename, sep="\t", index=False, header=True, na_rep="n/a")
        filename = self._get_bids_filename("simulation.json")
        with open(filename, "w") as f:
            json.dump(self.simulation_sidecar, f, indent=4)
        filename = self._get_bids_filename(f"internals.{self.fileformat}")    
        #self.internals.write_edf(filename)

        channel_names = [f"{i}" for i in range(self.internals.shape[0])]
        signal_headers = make_signal_headers(
            channel_names, 
            sample_frequency=self.emg_sidecar["SamplingFrequency"]
        )
        write_edf(filename, self.internals, signal_headers)


    def read(self):
        """Override read method to include simulated data"""
        # Call parent's read method first
        super().read()

        # Read simulation-specific files
        filename = self._get_bids_filename("spikes.tsv")
        if os.path.isfile(filename):
            self.spikes = pd.read_table(filename, on_bad_lines="warn")
        filename = self._get_bids_filename("motorunits.tsv")    
        if os.path.isfile(filename):
            self.motor_units = pd.read_table(
                filename, on_bad_lines="warn"
            )
        filename = self._get_bids_filename("internals.tsv")    
        if os.path.isfile(filename):
            self.internals_sidecar = pd.read_table(
                filename, on_bad_lines="warn"
            )
        filename = self._get_bids_filename("simulation.json")    
        if os.path.isfile(filename):
            with open(filename, "r") as f:
                self.simulation_sidecar = json.load(f)
        filename = self._get_bids_filename(f"internals.{self.fileformat}")
        if os.path.isfile(filename):
            #self.internals = read_edf(filename)
            self.internals, _, _ = read_edf(filename)



class bids_decomp_derivatives(bids_emg_recording):

    BIDSIGNORE = [
        "*sources.edf",
        "*sources.bdf",
        "*sources.json",
        "*log.json",
        "descriptions.tsv"
    ]

    INHERITABLE_FILES = ["events.json"]
    INHERITABLE_LEVELS = ["dataset" , "task", "subject", "session"]

    def __init__(
        self,
        pipelinename="pipelineName",
        format="standalone",
        parent_recording=None,
        datasetname="datasetName",
        datatype="emg",
        subject_label="01",
        session_label=None,
        task_label="isometric",
        acq_label=None,
        run_label="01",
        recording_label=None,
        desc_label="decomposed",
        inherited_metadata=None,
        inherited_level = None,
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
        #self.datapath = datapath
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
        self.datasetname = datasetname
        self.pipelinename = pipelinename

        # Adopt labels from an emg recording in BIDS format
        if isinstance(parent_recording, bids_emg_recording):
            root = str(Path(parent_recording.root).parent)
            datasetname = parent_recording.datasetname
            self.root = parent_recording.root
            self.datasetname = parent_recording.datasetname
            self.subject_label = parent_recording.subject_label
            self.session_label = parent_recording.session_label
            self.task_label = parent_recording.task_label
            self.acq_label = parent_recording.acq_label
            self.run_label = parent_recording.run_label
            self.recording_label = parent_recording.recording_label
            self.datatype = parent_recording.datatype

        # Make a BIDS compatible folder structure
        if format == "standalone":
            self.datasetname = f"{datasetname}-{pipelinename}"
            self.derivative_root = str(Path(root) / self.datasetname) + "/"
        else:
            self.datasetname = datasetname
            self.derivative_root = str(Path(root) / datasetname) + f"/derivatives/{pipelinename}/"

        self.root = str(Path(root) / self.datasetname) + "/"
        self.derivative_datapath = self.derivative_root + datapath
        self.format = format

        #self.source = Edf([EdfSignal(np.zeros(1), sampling_frequency=1)])
        self.fsamp = fsamp
        self.source = np.empty((0,))
        self.events = pd.DataFrame(columns=["onset", "duration", "sample", "unit_id", "description"])
        self.events_sidecar = self._init_event_sidecar()
        self.descriptions = pd.DataFrame(columns=["desc_id", "description"])
        self.descriptions.loc[0, "desc_id"] = "decomposed"
        self.descriptions.loc[0, "description"] = "estimated spiking motor unit activity"
        self.source_sidecar = {
            "SamplingFrequency": fsamp,
            "NumberOfSources": 0,
            "RecordingDuration": 0
        }
        name = f"{datasetname}-{pipelinename}" if format == "standalone" else f"{pipelinename} Outputs"
        self.dataset_sidecar = {
            "Name": name,
            "BIDSVersion": self._get_bids_version(),
            "DatasetType": "derivative",
            "GeneratedBy": [{
                "Name": pipelinename
            }]
        }
        self.log = {}

        self.code = []

        self.readme = """# Some BIDS derivative

        Your dataset description goes here

        """ 

        # Initialize empty inheritance dictionary
        self.inherited_metadata = {}
        self.inherited_levels = {}

        # Set inherited metadata if provided
        if inherited_metadata is not None:
            self._set_inherited_metadata(inherited_metadata, inherited_level)

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
            "description": {
                "Description": "Free text event description."
            }
        }

        return metadata
    
    def _get_bids_filename(self, extension):

        """
        Get a BIDS compatible filename
        
        """

        # Non inherited metdadata files
        if not self.inherited_metadata.get(extension, False): 

            folder = self.derivative_root
            if extension == "log.json":
                folder = f"{folder}logs/"        

            fname = f"sub-{self.subject_label}_"
            folder = f"{folder}sub-{self.subject_label}/" 

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

            fname = f"{fname}desc-{self.desc_label}_"       

        # Inherited metdadata files
        else:
            level = self.inherited_levels[extension]
            if level == "dataset":
                fname = ""
                folder = self.derivative_root
            elif level == "task":
                fname = self.task_label
                folder = self.derivative_root
            elif level == "subject":
                fname = f"sub-{self.subject_label}_"
                folder = f"{self.derivative_root}sub-{self.subject_label}/{self.datatype}/"
            elif level == "session":
                fname = f"sub-{self.subject_label}_ses-{self.session_label}_"
                folder = f"{self.derivative_root}sub-{self.subject_label}/ses-{self.session_label}/{self.datatype}"
            fname = f"{fname}desc-{self.desc_label}_"    
           
        if extension is None:
            name = folder + fname
        else:
            name = folder + fname + extension  

        return name

    def write(self):
        """
        Save dataset in BIDS format

        """
        # Generate an empty set of folders for your BIDS dataset
        if not os.path.exists(self.derivative_root):
            os.makedirs(self.derivative_root)
        if not os.path.exists(self.derivative_datapath):
            os.makedirs(self.derivative_datapath)
        
        # write *_events.tsv
        fname = self._get_bids_filename("events.tsv")
        self.events.to_csv(
            fname, sep="\t", index=False, header=True, na_rep="n/a"
        )
        # write events.json
        fname = self._get_bids_filename("events.json")
        if self.overwrite or not os.path.isfile(fname):
            with open(fname, "w") as f:
                json.dump(self.events_sidecar, f, indent=4)   
        # write *_log.json
        if self.log:  
            subfolder = self.derivative_datapath.split(self.derivative_root)[1]
            if not os.path.exists(f"{self.derivative_root}logs/{subfolder}"):   
                os.makedirs(f"{self.derivative_root}logs/{subfolder}")   
            fname = self._get_bids_filename("log.json")
            if self.overwrite or not os.path.isfile(fname):
                with open(fname, "w") as f:
                    json.dump(self.log, f, indent=4)   
        # write *_desc-decomposed_sources.edf + sidecar file
        if self.source.size > 0:   
            fname = self._get_bids_filename(f"sources.{self.fileformat}")
            channel_names = [f"Unit_{i}" for i in range(self.source.shape[0])]
            signal_headers = make_signal_headers(
                channel_names, 
                sample_frequency=self.fsamp
            )
            write_edf(fname, self.source, signal_headers)
            fname = self._get_bids_filename("sources.json")
            with open(fname, "w") as f:
                json.dump(self.source_sidecar, f, indent=4)
        # write dataset.json
        fname = f"{self.derivative_root}dataset_description.json"
        if self.overwrite or not os.path.isfile(fname):
            with open(fname, "w") as f:
                json.dump(self.dataset_sidecar, f, indent=4)   
        # write descriptions.tsv
        fname = f"{self.derivative_root}descriptions.tsv" 
        if self.overwrite or not os.path.isfile(fname):
            self.descriptions.to_csv(
                fname, sep="\t", index=False, header=True, na_rep="n/a"
            )
        # write README
        if self.format == "standalone":
            name = f"{self.derivative_root}README.md"
            if self.overwrite or not os.path.isfile(name):
                with open(name, "w", encoding="utf-8") as f:
                    f.write(self.readme)      
        # write .bidsignore
        if len(self.BIDSIGNORE) > 0:
            fname = Path(self.root) / ".bidsignore"
            if self.overwrite or not fname.exists():
                fname.write_text("\n".join(self.BIDSIGNORE) + "\n")
        # Save code files
        if len(self.code) > 0:  
            if not os.path.exists(self.derivative_root + "code/"):   
                os.makedirs(self.derivative_root + "code/")   
            for i in range(len(self.code)):
                if os.path.isfile(self.code[i]):
                    shutil.copy(self.code[i], f"{self.derivative_root}code/")
                
                   

    def read(self):
        """
        Import data from BIDS dataset

        """             

        # read *_predictedspikes.tsv
        fname = self._get_bids_filename("events.tsv")
        if os.path.isfile(fname):
            self.events = pd.read_table(fname, on_bad_lines="warn")
        # read *_events.json
        fname = self._get_bids_filename("events.json")
        if os.path.isfile(fname):
            with open(fname, "r") as f:
                self.events_sidecar = json.load(f)    
        # read *_sources.json
        fname = self._get_bids_filename("sources.json")
        if os.path.isfile(fname):
            with open(fname, "r") as f:
                self.source_sidecar = json.load(f)
        # read *_log.json
        fname = self._get_bids_filename("log.json")
        if os.path.isfile(fname):
            with open(fname, "r") as f:
                self.source_sidecar = json.load(f)        
        # read *.edf file
        fname = self._get_bids_filename(f"sources.{self.fileformat}")
        if os.path.isfile(fname):
            #self.source = read_edf(fname)
            self.source, _, _ = read_edf(fname)
        # read dataset.json
        fname = f"{self.derivative_root}dataset_description.json"
        if os.path.isfile(fname):
            with open(fname, "r") as f:
                self.dataset_sidecar = json.load(f)          
        # read descriptions.tsv
        fname = f"{self.derivative_root}descriptions.tsv" 
        if os.path.isfile(fname):
            self.descriptions = pd.read_table(fname, on_bad_lines="warn")  
        # read README.md
        name = f"{self.derivative_root}README.md"
        if os.path.isfile(name):
            with open(name, "r", encoding="utf-8") as f:
                self.readme = f.read()      
        # read .bidsignore   
        fname = Path(self.derivative_root) / ".bidsignore"
        if fname.exists():    
            self.BIDSIGNORE = fname.read_text(encoding="utf-8").splitlines()
        # list file names in subfolder code    
        code_path = Path(f"{self.derivative_root}code/")
        if code_path.exists:
            files = list(code_path.rglob("*"))
            self.code = [f.name for f in files]               

    def add_spikes(self, spikes, fsamp):
        """
        Convert a dictionary of spike times to long-format TSV-style DataFrame.

        Parameters:
            spikes (dict or DataFrame): {source_id: list of spike timings (in samples)} 
                    or long-format table (must contain unit_id and one of onset or sample)
            fsamp(float): Sampling frequency in Hz

        """

        if isinstance(spikes, dict):
            rows = []
            for unit_id, spike_times in spikes.items():
                for t in spike_times:
                    rows.append({
                        "onset": t/fsamp,
                        "duration": 0,
                        "sample": t,
                        "unit_id": unit_id, 
                        "description": "motor-unit-spike"
                    })

            frames = [self.events, pd.DataFrame(rows)]
            
        elif isinstance(spikes, pd.DataFrame):

            if "onset" not in spikes.columns:
                spikes["onset"] = spikes["sample"] / fsamp
            if "sample" not in spikes.columns:
                spikes["sample"] = spikes["onset"] * fsamp      
            if "duration" not in spikes.columns:
                spikes["duration"] = 0
            if "description" not in spikes.columns:
                spikes["description"] =  "motor-unit-spike"

            spikes = spikes.loc[:, ["onset", "duration", "sample", "unit_id", "description"]]      
            frames = [self.events, spikes] 

        frames = [f for f in frames if not f.empty]
        self.events = pd.concat(frames, ignore_index=True)
        self.events = self.events.drop_duplicates(subset=["onset", "unit_id", "sample"])
        self.events = self.events.sort_values(by=["onset"])

    def set_data(self, field_name, mydata, fsamp):

        super().set_data(field_name, mydata, fsamp)

        if field_name == "source":
            self.fsamp = fsamp
            self.source_sidecar["SamplingFrequency"] = fsamp
            self.source_sidecar["NumberOfSources"] = mydata.shape[0]
            self.source_sidecar["RecordingDuration"] = mydata.shape[1] / fsamp

def run_bids_validator(
        path,
        ignored_codes = [],
        ignored_fields = [],
        ignored_files = [],
        print_errors=True, 
        print_warnings=True
    ):
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
        valid (bool): True if there are no errors  
    
    """

    # Run bids validator
    result = subprocess.run(
        ["bids-validator-deno", "--format", "json", path],
        capture_output=True,
        text=True
    )
    # Extract and filter all issues
    validation = json.loads(result.stdout)
    issues = validation["issues"]["issues"]
    issues = [f for f in issues if (not "code" in f or f["code"] not in ignored_codes)]
    issues = [f for f in issues if (not "subCode" in f or f["subCode"] not in ignored_fields)]
    issues = [f for f in issues if (not "location" in f or f["location"] not in ignored_files)]
    # Split issues in errors and warnings
    errors = [f for f in issues if f["severity"] == "error"]
    warnings = [f for f in issues if f["severity"] == "warning"]
    # Print issues
    if print_errors:
        print(f"Number of detected errors: {len(errors)}")
        print(json.dumps(errors, indent=4))
    if print_warnings:    
        print(f"Number of detected warnings: {len(warnings)}")
        print(json.dumps(warnings, indent=4))
    # Check if the folder is BIDS valid
    valid = True if len(errors) == 0 else False    

    return errors, warnings, valid

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
