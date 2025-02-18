"""
File that calculates each subject connectivity graph and its metrices in two connectivity regions:

- gray-to-gray matter (GM-GM)
- white-to-white matter (WM-WM)

Atlas GM: Neuromorphometrics (NMM)
Atlas WM: John Hopkins University (JHU, https://identifiers.org/neurovault.collection:264)

Graph are calculated by keeping only the significant positive edges of the FC, using wilcoxon signed-rank tests and 
correction for multiple comparisons with fdr_bh


Graph metrics calculated: 
- Average weighted node strength
- Average weighted clustering coefficient
- Weighted global efficiency¹

¹Formula from Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: Uses and interpretations. 
NeuroImage, 52 (3), 1059–1069. https://doi.org/10.1016/j.neuroimage.2009.10.003
"""

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from copy import deepcopy
import statannot
import sys
import os
sys.path.insert(1, os.path.dirname(sys.path[0]))
from utils import common_regions, load_FC, wilcoxon_sign_edges
from fig_style import fig_size
import json

fieldstrength = ["3T", "7T"]

params_path = 'params.json'

# Read and parse the JSON file
with open(params_path, 'r') as file:
    params = json.load(file)

data_folder  = Path(f'{params["paths"]["data_path"]}')
savefig_path = params["paths"]["savefig_path"]

# NMM-NMM = GM-GM
# JHU-JHU = WM-WM
# Cannot do "NMM-JHU" cause non-squared matrix
ROIs        = ["NMM-NMM FC", "JHU-JHU FC"]  
# "se_rs" (BOLD), "diff_rs" (ADC-fMRI)
contrasts   = ["se_rs", "diff_rs"]
# percentage of subjects in which the ROI should be present
perc_common = 1
network_metrices = pd.DataFrame(
    columns=["Metric", "Metric (weighted)", "data", "contrast", "field", "region"]
)
# "NMM" = neuromorphometric, gray matter atlas
GM_region = "NMM" 
GSR       = True

for field in fieldstrength:

    FC_name = f"FC_{GM_region}_JHU_{GM_region}_JHU_GSR_only_all_{field}_10vx_reg.npy"

    # All the ROIs
    GM_all_ROIs, WM_all_ROIs = common_regions(field, 0, data_folder)
    # Only the ROIs that are present in all the subjects and all contrasts
    FC_all_contrasts_all_subjects, GM_common_ROIs, WM_common_ROIs = load_FC(
        perc_common, GM_all_ROIs, WM_all_ROIs, FC_name, data_folder
    )
    GM_common_ROIs    = list(GM_common_ROIs)
    WM_common_ROIs    = list(WM_common_ROIs)
    GM_WM_common_ROIs = GM_common_ROIs + WM_common_ROIs

    for ROI in ROIs:
        for idx, contrast in enumerate(contrasts):

            FC_all = deepcopy(FC_all_contrasts_all_subjects[idx])
            
            mask = wilcoxon_sign_edges(FC_all)
            if not np.allclose(mask.T, mask):
                mask = mask + mask.T
            mask  = np.logical_and(mask, ~np.identity(mask.shape[0]).astype('bool'))

            for subject_id, subject in enumerate(FC_all):
                if ROI == "NMM-NMM FC":
                    FC_subject = deepcopy(np.arctanh(FC_all[subject_id, ...]))
                    FC_subject[~mask] = 0
                    FC_subject = FC_subject[
                        : len(GM_common_ROIs), : len(GM_common_ROIs)
                    ]
                elif ROI == "JHU-JHU FC":
                    FC_subject = deepcopy(np.arctanh(FC_all[subject_id, ...]))
                    FC_subject[~mask] = 0
                    FC_subject = FC_subject[
                        len(GM_common_ROIs) :, len(GM_common_ROIs) :
                    ]

                FC_subject[np.isnan(FC_subject)] = 0
                FC_subject[FC_subject < 0]       = 0

                FC_subject_df = pd.DataFrame(FC_subject)
                
                if ROI == "NMM-NMM FC":
                    FC_subject_df.columns = GM_common_ROIs
                    FC_subject_df.index   = GM_common_ROIs
        
                elif ROI == "JHU-JHU FC":
                    FC_subject_df.columns = WM_common_ROIs
                    FC_subject_df.index   = WM_common_ROIs

                G_pos = nx.from_pandas_adjacency(FC_subject_df)
                G_pos.remove_nodes_from(list(nx.isolates(G_pos)))

                # Weighted average node strength
                degrees = G_pos.degree(weight='weight')
                average_node_strength = 0
                for node in degrees:
                    average_node_strength = average_node_strength + node[1]
                average_node_strength = average_node_strength / len(G_pos)               

                network_metrices.loc[len(network_metrices)] = {
                    "Metric (weighted)": "Average node strength",
                    "data": average_node_strength,
                    "contrast": contrast,
                    "field": field,
                    "region": ROI,
                }

                # Weighted average clustering
                network_metrices.loc[len(network_metrices)] = {
                    "Metric (weighted)": "Average clustering",
                    "data": nx.average_clustering(G_pos, weight="weight"),
                    "contrast": contrast,
                    "field": field,
                    "region": ROI,
                }
                
                # Weighted global efficiency
                avg_all_nodes = 0
                G_path        = nx.from_pandas_adjacency(1 / FC_subject_df)
                nodes         = G_path.nodes
                shortest_path_length = dict(nx.shortest_path_length(G_path, weight="weight"))
                for entry in shortest_path_length:
                    avg_per_node = 0
                    existing_links = dict(nx.all_pairs_dijkstra_path_length(G_path, weight="weight"))[entry]
                    for entry2 in nodes:
                        if entry != entry2:
                            if entry2 in existing_links:
                                avg_per_node = avg_per_node + 1/existing_links[entry2]
                    avg_per_node  = avg_per_node / (len(nodes) - 1)
                    avg_all_nodes = avg_all_nodes + avg_per_node
                avg_all_nodes = avg_all_nodes / len(nodes)

                network_metrices.loc[len(network_metrices)] = {
                    "Metric (weighted)": "Global efficiency",
                    "data": avg_all_nodes,
                    "contrast": contrast,
                    "field": field,
                    "region": ROI,
                }
 
# # Store
# if GSR:
#     network_metrices.to_csv(data_folder / "weighted_metrics_GSR.csv", index=False)
# else:
#     network_metrices.to_csv(data_folder / "weighted_metrics_no_GSR.csv", index=False)

# Load
if GSR:
    network_metrices = pd.read_csv(data_folder / "weighted_metrics_GSR.csv")
else:
    network_metrices = pd.read_csv(data_folder / "weighted_metrics_no_GSR.csv")


for field in fieldstrength:
    for ROI in ROIs:
        fig, ax = plt.subplots(1, 3, figsize=fig_size(.75, 0.4))
        net_per_field = network_metrices[
            (network_metrices["field"] == field) & (network_metrices["region"] == ROI)
        ]

        for idx, n in enumerate(net_per_field["Metric (weighted)"].unique()):
            bold_val = net_per_field[(net_per_field["Metric (weighted)"] == n) & (net_per_field.contrast == "se_rs")].data.values
            diff_val = net_per_field[(net_per_field["Metric (weighted)"] == n) & (net_per_field.contrast == "diff_rs")].data.values

            my_pal = np.array(sns.color_palette("colorblind"))[[0, 3]]
            net_no_assort = net_per_field[net_per_field["Metric (weighted)"] == n]
            g = sns.violinplot(
                               data=net_no_assort,
                               y="data",
                               x="Metric (weighted)",
                               hue="contrast",
                               hue_order=contrasts,
                               ax=ax[idx],
                               xlab="",
                               palette=my_pal,
                               dodge=True,
                               legend=False
                               )

            sns.stripplot(
                          x="Metric (weighted)", 
                          y="data", 
                          data=net_no_assort, 
                          hue="contrast", 
                          hue_order=contrasts,
                          jitter=True, 
                          zorder=1, 
                          ax=ax[idx], 
                          dodge=True, 
                          palette=my_pal,
                          edgecolor='darkgray',
                          linewidth=1,
                          alpha=0.8
                          )
            
            for collection in ax[idx].collections:
                if isinstance(collection, matplotlib.collections.PolyCollection):
                    collection.set_edgecolor(collection.get_facecolor())
                    collection.set_facecolor(collection.get_facecolor())
                    collection.set_alpha(0.1)

            handles, labels = ax[idx].get_legend_handles_labels()

            # Statistical tests
            couples = []
            for cat in net_no_assort.contrast.unique():
                for met in net_no_assort["Metric (weighted)"].unique():
                    if not net_no_assort[
                        (net_no_assort.contrast == cat) & (net_no_assort["Metric (weighted)"] == met)
                    ].empty:
                        couples.append((met, cat))

            couples_end = []
            for met in net_no_assort["Metric (weighted)"].unique():
                for i in range(len(couples)):
                    for j in range(i + 1, len(couples)):
                        if (
                            (couples[i][0] == met)
                            and (couples[j][0] == met)
                        ):
                            couples_end.append((couples[i], couples[j]))

            statannot.add_stat_annotation(
                                          ax[idx],
                                          data=net_no_assort,
                                          y="data",
                                          x="Metric (weighted)",
                                          hue="contrast",
                                          hue_order=contrasts,
                                          box_pairs=couples_end,
                                          test="Mann-Whitney",
                                          text_format="star",
                                          loc="inside",
                                          )

            # When creating the legend, only use the first two elements
            # to effectively remove the last two.
            ax[idx].legend([], [], frameon=False)
                
            ax[idx].tick_params(bottom=False, labelbottom=False)
            ax[idx].set(ylabel=n, 
                        xlabel=None, 
                        ylim=[net_no_assort.data.min() - .6 * (net_no_assort.data.max() - net_no_assort.data.min()), 
                              net_no_assort.data.max() + .6 * (net_no_assort.data.max() - net_no_assort.data.min())],
                        xlim=[-0.5, 0.5])

            ROI_to_write = ROI.replace(" ", "_").replace("-", "_")
            print("Saved in : ", f"{savefig_path}/general_metrics_weighted_{ROI_to_write}_{field}.png")
            fig.savefig(f"{savefig_path}/general_metrics_weighted_{ROI_to_write}_{field}.png")
            fig.savefig(f"{savefig_path}/general_metrics_weighted_{ROI_to_write}_{field}.pdf")
            
        