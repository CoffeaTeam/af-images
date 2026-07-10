import importlib.util
import os

import awkward as ak
import dask
import pytest
from coffea import processor
from coffea.nanoevents import BaseSchema
from coffea.nanoevents.methods import candidate
from dask.distributed import Client

fileset = (
    "https://github.com/CoffeaTeam/coffea/raw/master/tests/samples/nano_dimuon.root"
)

# Capability detection: what is actually importable in this image?
HAS_DASK_AWKWARD = all(
    importlib.util.find_spec(m) for m in ("dask_awkward", "dask_histogram")
)

# What the image *claims* it has, baked in at build time by the Dockerfile.
_DECLARED = os.environ.get("AF_DASK_AWKWARD")

requires_dask_awkward = pytest.mark.skipif(
    not HAS_DASK_AWKWARD, reason="image built without dask-awkward"
)


@pytest.mark.v0
@pytest.mark.calver
@pytest.mark.skipif(_DECLARED is None, reason="AF_DASK_AWKWARD not set by image")
def test_image_matches_declared_capability():
    """Guard against a solver regression pulling dask-awkward in or out."""
    assert HAS_DASK_AWKWARD == (_DECLARED == "1")


class MyProcessor(processor.ProcessorABC):
    def __init__(self, mode):
        assert mode in ["eager", "virtual", "dask"]
        self._mode = mode

    def process(self, events):
        dataset = events.metadata["dataset"]
        muons = ak.zip(
            {
                "pt": events.Muon_pt,
                "eta": events.Muon_eta,
                "phi": events.Muon_phi,
                "mass": events.Muon_mass,
                "charge": events.Muon_charge,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

        if self._mode == "dask":
            from hist.dask import Hist as hist_class
        else:
            from hist import Hist as hist_class

        h_mass = (
            hist_class.new.StrCat(["opposite", "same"], name="sign")
            .Log(1000, 0.2, 200.0, name="mass", label=r"$m_{\mu\mu}$ [GeV]")
            .Int64()
        )

        cut = (ak.num(muons) == 2) & (ak.sum(muons.charge, axis=1) == 0)
        # add first and second muon in every event together
        dimuon = muons[cut][:, 0] + muons[cut][:, 1]
        h_mass.fill(sign="opposite", mass=dimuon.mass)

        cut = (ak.num(muons) == 2) & (ak.sum(muons.charge, axis=1) != 0)
        dimuon = muons[cut][:, 0] + muons[cut][:, 1]
        h_mass.fill(sign="same", mass=dimuon.mass)

        return {
            dataset: {
                "entries": ak.num(events.run, axis=0),
                "mass": h_mass,
            }
        }

    def postprocess(self, accumulator):
        pass


@pytest.mark.v0
def test_processor_dimu_massv0():
    with Client() as client:
        executor = processor.DaskExecutor(client=client)
        run = processor.Runner(
            executor=executor,
            schema=BaseSchema,
            chunksize=20,
        )
        out = run(
            {"dimuon": [fileset]},
            treename="Events",
            processor_instance=MyProcessor("virtual"),
        )
        print(out)
        assert out["dimuon"]["entries"] == 40


@pytest.mark.calver
def test_dimu_mass_runner():
    """Needs distributed only -- runs in both image variants."""
    with Client() as client:
        run = processor.Runner(
            executor=processor.DaskExecutor(client=client),
            schema=BaseSchema,
            chunksize=20,
        )
        out = run(
            {"DoubleMuon": {"files": {fileset: "Events"}}},
            processor_instance=MyProcessor("virtual"),
        )
        assert out["DoubleMuon"]["entries"] == 40


@pytest.mark.calver
@requires_dask_awkward
def test_dimu_mass_dask():
    """Needs hist.dask -> dask_histogram. Skipped on the experimental image."""
    from coffea.dataset_tools import apply_to_fileset, preprocess

    with Client() as client:
        dataset_runnable, _ = preprocess(
            {"DoubleMuon": {"files": {fileset: "Events"}}},
            step_size=20,
            align_clusters=False,
            files_per_batch=1,
            skip_bad_files=True,
            save_form=False,
            scheduler=client,
        )
        to_compute = apply_to_fileset(
            MyProcessor("dask"),
            dataset_runnable,
            schemaclass=BaseSchema,
        )
        (out,) = dask.compute(to_compute)
        assert out["DoubleMuon"]["entries"] == 40