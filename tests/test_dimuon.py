import dask
import awkward as ak

from coffea import processor
from coffea.nanoevents.methods import candidate
from coffea.nanoevents import BaseSchema, NanoEventsFactory, NanoAODSchema

from dask.distributed import Client
import pytest

fileset = "https://github.com/CoffeaTeam/coffea/raw/master/tests/samples/nano_dimuon.root"

class MyProcessor(processor.ProcessorABC):
    def __init__(self, mode):
        assert mode in ["eager", "virtual", "dask"]
        self._mode = mode

    def process(self, events):
        dataset = events.metadata['dataset']
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
            hist_class.new
            .StrCat(["opposite", "same"], name="sign")
            .Log(1000, 0.2, 200., name="mass", label="$m_{\mu\mu}$ [GeV]")
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
        run = processor.Runner(executor=executor,
                            schema=BaseSchema)
        out = run({"dimuon": [fileset]},
                treename="Events",
                processor_instance=MyProcessor("virtual"))
        print(out)
        assert out["dimuon"]["entries"] == 40

@pytest.mark.calver
def test_dimu_masscalver():
    with Client() as client:
        executor = processor.DaskExecutor(client=client)
        run = processor.Runner(executor=executor,
                            schema=BaseSchema)
        out = run({"DoubleMuon": {"files": {fileset: "Events"}}},
                    processor_instance=MyProcessor("virtual"))
        print(out)
        assert out["DoubleMuon"]["entries"] == 40

        events = NanoEventsFactory.from_root(
            {fileset: "Events"},
            metadata={"dataset": "DoubleMuon"},
            schemaclass=BaseSchema,
            mode="dask",
            ).events()
        p = MyProcessor("dask")
        out = p.process(events)
        (computed,) = dask.compute(out)
        print(computed)
        assert computed["DoubleMuon"]["entries"] == 40
    #from coffea.dataset_tools import (
    #    apply_to_fileset,
    #    max_chunks,
    #    preprocess,
    #)
    #Still not sure where it is used (?)
    #client = Client()
    #dataset_runnable = preprocess(
    #    fileset,
    #    align_clusters=False,
    #    files_per_batch=10,
    #    skip_bad_files=True,
    #    save_form=False,
    #)
    #to_compute = apply_to_fileset(
    #            MyProcessor(),
    #            max_chunks(dataset_runnable, 300),
    #            schemaclass=BaseSchema,
    #        )
    #(out,) = dask.compute(to_compute)
