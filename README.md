# Docker images for Coffea Columnar Object Framework For Effective Analysis


[![GitHub issues](https://img.shields.io/github/issues/coffeateam/af-images)](https://github.com/CoffeaTeam/af-images/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/coffeateam/af-images)](https://github.com/CoffeaTeam/af-images/pulls)

Latest DockerHub Images: https://hub.docker.com/orgs/coffeateam/repositories

# Coffea image tags

- `dev-` tags represent the status of `dev` branch
- `head-` tags represent the status of `main` branch
- `202x.x.x-` and `0.7.x.-`, as well `latest` represent relevant tags in [this repository](https://github.com/CoffeaTeam/af-images/tags)

# Image flavours

Images are named `<flavour>-<distro>`. The flavour says what is in the image:

| Flavour      | coffea   | `dask-awkward` / `dask-histogram` | `dask` / `distributed` |
| ------------ | -------- | --------------------------------- | ---------------------- |
| `coffea-0.7` | 0.7.x    | n/a                               | as resolved            |
| `coffea-dak` | CalVer   | yes (`dak` is the usual import alias) | as pinned in the env file |
| `coffea`     | CalVer   | no                                | no upper pin           |

Use `coffea-dak-*` unless you have a reason not to: it is the supported image
and the one that can run the `dask-awkward` / `hist.dask` code paths.

`coffea-*` exists to test whether dropping `dask-awkward` allows a newer
`dask` / `distributed`. It cannot run `hist.dask`.

# Coffea 2024+ (CalVer) images

- Dockerfiles and conda environment files are available in `coffea-dask` directory
- All distro variants build from a single `coffea-dask/Dockerfile`, selected by
  the `BASE_IMAGE`, `ENV_YAML` and `NODAK` build args

| Image                       | Environment file          | Description                                                                 | Last Tag Version |
| --------------------------- | ------------------------- | --------------------------------------------------------------------------- | ---------------- |
| `coffea-dak-almalinux8`     | `environment.yaml`        | Alma8, CalVer coffea with dask-awkward, latest XRootD and CA certificates    | [![Docker Image Version](https://img.shields.io/docker/v/coffeateam/coffea-dak-almalinux8)](https://hub.docker.com/r/coffeateam/coffea-dak-almalinux8/tags) |
| `coffea-dak-almalinux9`     | `environment.yaml`        | Alma9, CalVer coffea with dask-awkward, latest XRootD and CA certificates    | [![Docker Image Version](https://img.shields.io/docker/v/coffeateam/coffea-dak-almalinux9)](https://hub.docker.com/r/coffeateam/coffea-dak-almalinux9/tags) |
| `coffea-dak-almalinux9-eaf` | `environment-eaf.yaml`    | Alma9, EAF variant                                                           | [![Docker Image Version](https://img.shields.io/docker/v/coffeateam/coffea-dak-almalinux9-eaf)](https://hub.docker.com/r/coffeateam/coffea-dak-almalinux9-eaf/tags) |
| `coffea-almalinux9`         | `environment.yaml` (filtered) | Alma9, same env as `coffea-dak-almalinux9` with dask-awkward removed and the dask pin lifted | [![Docker Image Version](https://img.shields.io/docker/v/coffeateam/coffea-almalinux9)](https://hub.docker.com/r/coffeateam/coffea-almalinux9/tags) |
| `coffea-almalinux9-noml`    | `environment-noml.yaml`   | Alma9 without the ML stack. Has no dask-awkward and no dask pin              | [![Docker Image Version](https://img.shields.io/docker/v/coffeateam/coffea-almalinux9-noml)](https://hub.docker.com/r/coffeateam/coffea-almalinux9-noml/tags) |

> `environment-eaf.yaml` currently lists `dask-awkward` but not
> `dask-histogram`, so `hist.dask` is not importable in the EAF image and the
> `hist.dask` test is skipped there.

# Coffea 0.7.x images

- Dockerfiles and conda environment files are available in `coffea-base` directory

| Image                   | Description                                                        | Last Tag Version |
| ----------------------- | ------------------------------------------------------------------ | ---------------- |
| `coffea-0.7-almalinux8` | Alma8 Dask Coffea 0.7.x image with latest XRootD and CA certificates | [![Docker Image Version](https://img.shields.io/docker/v/coffeateam/coffea-0.7-almalinux8)](https://hub.docker.com/r/coffeateam/coffea-0.7-almalinux8/tags) |
| `coffea-0.7-almalinux9` | Alma9 Dask Coffea 0.7.x image with latest XRootD and CA certificates | [![Docker Image Version](https://img.shields.io/docker/v/coffeateam/coffea-0.7-almalinux9)](https://hub.docker.com/r/coffeateam/coffea-0.7-almalinux9/tags) |

# Renamed images

The previous names said which *directory* an image was built from, not what was
in it: `dask` meant CalVer and `base` meant 0.7.x. They have been renamed:

| Old name                       | New name                   | Contents changed? |
| ------------------------------ | -------------------------- | ----------------- |
| `coffea-base-almalinux8`       | `coffea-0.7-almalinux8`    | no                |
| `coffea-base-almalinux9`       | `coffea-0.7-almalinux9`    | no                |
| `coffea-dask-almalinux8`       | `coffea-dak-almalinux8`    | no                |
| `coffea-dask-almalinux9`       | `coffea-dak-almalinux9`    | no                |
| `coffea-dask-almalinux9-eaf`   | `coffea-dak-almalinux9-eaf`| no                |
| `coffea-dask-almalinux9-noml`  | `coffea-almalinux9-noml`   | no (it never had dask-awkward) |

The old names are still published as aliases of the new ones — same digest,
same contents — so existing pulls keep working. **They will stop being updated;
please move to the new names.** No old name has been reused for a different
image, so nothing you already pull can silently change underneath you.

# Registries

Images are published to:

- Docker Hub: `coffeateam/<image>`
- OSG Harbor: `hub.opensciencegrid.org/coffea-casa/<image>`
