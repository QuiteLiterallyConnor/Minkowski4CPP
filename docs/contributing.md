---
title: Contributing & credits
description: Repository structure, documentation maintenance, and project attribution.
permalink: /contributing/
---

## Repository map

| Directory | Responsibility |
|---|---|
| `MinkowskiEngine/` | Public sparse API and core implementation |
| `MinkowskiEngine/modules/` | Neural-network modules |
| `MinkowskiEngine/autograd/` | Differentiable operation bindings |
| `MinkowskiEngine/adaptive/` | Mixed-resolution geometry, topology, and feature operations |
| `MinkowskiEngine/gaussian/` | Gaussian geometry, scene tensors, renderer, and trainer |
| `tests/adaptive/`, `tests/gaussian/` | Focused subsystem tests |
| `tests/utils/` | Shared test utilities |
| `tests/builders/` | Builder and NVTX regression fixtures |
| `benchmarks/` | Benchmark sources and local JSON results |
| `scripts/` | Native build helpers |
| `apt/`, `docker/` | Packaging and container builders |
| `cmake/` | Installed package configuration |
| `docs/` | This GitHub Pages documentation |

## Making a change

1. Read the relevant public header and existing tests.
2. Keep changes scoped to the requested behavior.
3. Preserve feature dtype/device, coordinate identity, and autograd contracts.
4. Add focused regression coverage and run the smallest relevant targets.
5. Update the guide and API entry when user-facing behavior changes.
6. Report real build/runtime validation separately from mock or syntax checks.

Place Gaussian implementation in its subsystem, subsystem tests in their
test directories, and reusable test helpers in `tests/utils/`.
Do not commit private datasets, credentials, build artifacts, or machine
specific dependency paths.

## Documentation site

The site is plain Jekyll Markdown with a repository-owned layout and CSS.
There is no remote theme or client-side framework dependency.
Navigation is defined in `docs/_data/navigation.yml`; page metadata uses
YAML front matter. Give each guide a stable `permalink`.

Use relative Markdown links between guides. Jekyll/GitHub Pages resolves
them to published page URLs. For repository sources, use
`site.repository_url` rather than linking outside the Pages build directory.
Layout assets and navigation use `relative_url` so the project base path
works correctly.

The site supports narrow screens, keyboard navigation, and the user's
light/dark preference.

## Publish on GitHub Pages

The repository's Pages workflow:

1. Checks out the default branch.
2. Configures Pages metadata.
3. Builds `docs/` with the official Jekyll Pages action.
4. Uploads the generated static site.
5. Deploys it to the `github-pages` environment.

In **Settings > Pages > Build and deployment**, select **GitHub Actions**.
Push documentation changes to `main`, or run the workflow manually.
Pages must be enabled in repository settings; adding a workflow alone
cannot change that setting.

Expected project URL:
[quiteliterallyconnor.github.io/Minkowski4CPP](https://quiteliterallyconnor.github.io/Minkowski4CPP/).
If deploying elsewhere, update `url` / `baseurl` in `docs/_config.yml`.

To preview using the same build environment:

```bash
docker run --rm \
  --workdir /github/workspace \
  -e GITHUB_WORKSPACE=/github/workspace \
  -v "$PWD/docs:/github/workspace/docs" \
  -v "$PWD/_site:/github/workspace/_site" \
  -e INPUT_SOURCE=docs \
  -e INPUT_DESTINATION=_site \
  ghcr.io/actions/jekyll-build-pages:latest
```

Serve the generated directory with a local static HTTP server. The default
`baseurl` is `/Minkowski4CPP`, so serve it under that path or override the
base path for a root-level local preview.

## Credits and citation

Minkowski4CPP is a derivative of
[NVIDIA MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine).
The original sparse-convolution work is:

> Christopher Choy, JunYoung Gwak, and Silvio Savarese.
> *4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks*.
> CVPR, 2019, pp. 3075-3084.

```bibtex
@inproceedings{choy20194d,
  title={4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks},
  author={Choy, Christopher and Gwak, JunYoung and Savarese, Silvio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3075--3084},
  year={2019}
}
```

## License

The repository's [LICENSE]({{ site.repository_url }}/blob/main/LICENSE)
contains the MIT license. Preserve that notice and the upstream/third-party
licenses accompanying derivative code. The README also directs readers to
the original MinkowskiEngine terms; the top-level license does not erase
those obligations.
