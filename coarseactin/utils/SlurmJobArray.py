import os
import warnings
import itertools
import pandas
import time
import typing
from pathlib import Path


class SlurmJobArray:
    """
        Selects a single condition from an array of parameters using the SLURM_ARRAY_TASK_ID environment variable.
        The parameters need to be supplied as a dictionary. If the task is not in a slurm environment,
        the test parameters will supersede the parameters.
        ...
        Attributes
        ----------
            test_run: Boolean: This simulation is a test
            job_id: SLURM_ARRAY_TASK_ID
            all_parameters: Parameters used to initialize the job
            parameters: Parameters for this particular job
            name: The name (and relative path) of the output
        Methods
        -------
            print_parameters:
            print_slurm_variables:
            write_csv:
    """

    def __init__(self, name: str,
                 parameters: dict,
                 test_parameters: dict = None,
                 job_id: typing.Union[int, str] = None):
        """

        example:
            sjob = SlurmJobArray("ActinSimv6", parameters, test_parameters)
        Parameters
        ----------
        name: str
        parameters: dict
            Example:
                parameters={"epsilon":[100],
                            "aligned":[True,False],
                            "actinLen":[20,40,60,80,100,120,140,160,180,200,220,240,260,280,300],
                            "repetition":range(5),
                            "temperature":[300],
                            "system2D":[False],
                            "simulation_platform":["OpenCL"]}
        test_parameters: dict, Optional
            example:
                test_parameters={"simulation_platform":"CPU"}
        job_id: int | str
        """
        name = Path(name)
        self.all_parameters = parameters
        self.test_parameters = test_parameters
        if test_parameters is None:
            self.test_parameters = {}

        # Parse the slurm variables
        self.slurm_variables = {}
        for key in os.environ:
            if len(key.split("_")) > 1 and key.split("_")[0] == 'SLURM':
                self.slurm_variables.update({key: os.environ[key]})

        self.test_run = False
        self.command = None

        # Check if ran as slurm array and overwrite job id
        try:
            job_id = int(job_id)
        except ValueError:
            self.command = job_id
            job_id = 0
        except TypeError:
            job_id = None

        if job_id is None:
            try:
                job_id = int(self.slurm_variables["SLURM_ARRAY_TASK_ID"])
            except KeyError:
                warnings.warn("JobID not set")
                job_id = 0

        #Create directory if it does not exist
        parent = Path(name).parent
        try:
            parent.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"The directory {str(parent)} already exists")
        
        # Set job id
        self.job_id = job_id

        self.keys = parameters.keys()
        self.all_conditions = list(itertools.product(*[parameters[k] for k in self.keys]))

        extra_keys = self.test_parameters.keys()-self.keys
        if len(extra_keys) > 0:
            raise KeyError(f"Some keys in test_parameters is not set in parameters: {extra_keys}")

        self.root = name

        if self.command == 'jobs':
            self.write_jobs('slurm_jobs.txt')
            exit(1)
        elif self.command == 'test':
            warnings.warn("Test run will be executed")
            self.test_run = True

        # Assign the jobid name at least 3 characters or more depending in number of conditions
        self.job_name_size = max(3, len(str(len(self) - 1)))

    @property
    def parameter(self):
        return dict(zip(self.keys, self.all_conditions[self.job_id]))

    @property
    def job_name(self):
        if self.command:
            # If the job is a test, set "test" as the name
            return self.command
        else:
            return f"{self.job_id:0{self.job_name_size}d}"

    @property
    def name(self):
        """The name only includes enough information to differentiate the simulations."""
        name_out = f"{self.root}_{self.job_name}_"
        name_out += '_'.join([f"{a[0]}_{self[a]}" for a in self.parameter if len(self.all_parameters[a]) > 1])
        return name_out

    def __getitem__(self, name):
        if self.test_run:
            try:
                return self.test_parameters[name]
            except KeyError:
                return self.parameter[name]
        else:
            return self.parameter[name]

    def __getattr__(self, name: str):
        """ The keys of the parameters can be called as attributes
        """
        if name in self.__dict__:
            return object.__getattribute__(self, name)
        elif name in self.parameter:
            return self[name]
        else:
            return object.__getattribute__(self, name)

    def __repr__(self):
        return str(self.parameter)

    def __len__(self):
        return len(self.all_conditions)

    def keys(self):
        return str(self.parameters.keys())

    def print_parameters(self):
        print(f"Number of conditions: {len(self)}")
        print("Running Conditions")
        for k in self.keys:
            print(f"{k} :", f"{self[k]}")
        print()

    def print_slurm_variables(self):
        print("Slurm Variables")
        for key in self.slurm_variables:
            print(key, ":", self.slurm_variables[key])
        print()

    def write_csv(self, out=""):
        s = pandas.concat([pandas.Series({k: self[k] for k in self.keys}),
                           pandas.Series(self.slurm_variables)])
        s['test_run'] = self.test_run
        s['date'] = time.strftime("%Y_%m_%d")
        s['name'] = self.name
        s['job_id'] = self.job_id

        if out == '':
            s.to_csv(self.name + '.param')
        else:
            s.to_csv(out)

    def write_jobs(self, out=None, split=8):
        import inspect
        #print(inspect.stack())
        caller_path = inspect.stack()[2].filename
        job_list = []
        _job_id = self.job_id
        for job in range(len(self)):
            self.job_id = job
            job_list += [f"python {caller_path} {self.job_id} > {self.name}.log\n"]
        self.job_id = _job_id
        if out is None:
            return ''.join(job_list)
        else:
            lines_to_write = []
            for i, line in enumerate(job_list):
                lines_to_write += [line]
                if i % split == split - 1:
                    with open(f'{out}.{i // split}', 'w+') as out_file:
                        out_file.write(''.join(lines_to_write))
                    lines_to_write = []
            if len(lines_to_write)>0:
                with open(f'{out}.{i // split}', 'w+') as out_file:
                    out_file.write(''.join(lines_to_write))

        with open('slurm_run_array.sh', 'w+') as out_file:
            out_file.write(f"""#!/bin/bash
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=6
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:8
#SBATCH --array=0-{i // split}
#SBATCH --time=24:00:00
#SBATCH --export=ALL

module purge
module load  foss/2020b OpenMM Launcher_GPU

echo "My job is running on: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "my job started on: "
date

export LAUNCHER_WORKDIR=`pwd`
export LAUNCHER_JOB_FILE=slurm_jobs.txt.${{SLURM_ARRAY_TASK_ID}}
export LAUNCHER_BIND=1

$LAUNCHER_DIR/paramrun

echo "My job finished at:"
date""")


