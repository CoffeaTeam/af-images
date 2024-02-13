import time

import awkward as ak
import hist
import matplotlib.pyplot as plt
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor

import dask
import dask_awkward as dak
import hist.dask as hda

from distributed import Client
client=Client()

# The opendata files are non-standard NanoAOD, so some optional data columns are missing
NanoAODSchema.warn_missing_crossrefs = False

# The warning emitted below indicates steps_per_file is for initial data exploration
# and test. When running at scale there are better ways to specify processing chunks
# of files.
events = NanoEventsFactory.from_root(
    "root://eospublic.cern.ch//eos/root-eos/Run2012B_SingleMu.root:Events",
    steps_per_file=500,
    metadata={"dataset": "SingleMu"}
    ).events()


@pytest.mark.coffeacalver
def test_adl1():
    q1_hist = (
        hda.Hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]")
        .Double()
        .fill(events.MET.pt)
    )
    q1_hist.compute().plot1d(flow="none")
    dak.necessary_columns(q1_hist)
