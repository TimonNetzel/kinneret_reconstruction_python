"""Input contract of the Kinneret case (slot 3, add.sites <- T): what the bound code reads, where, and how far each input is bound.

Reads the frozen originals in diss by hash and refuses to write if a source or a pinned line differs. No R script, no model, no draw, no sampler.
The only evaluation of source values is the coordinate condition of mcmc_prepare.R:568-572, a min/max comparison on two columns of a preserved
grid file, kept apart under conditions/inside_giesecke and conditional on the older coordinates. Paths in the output are relative to diss.
Writes input_contract.json next to this file. Usage from the repository root: bind_input_contract.py
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from local_rdata_metadata import read_metadata  # noqa: E402
from local_rdata_stream import children, small_value  # noqa: E402

M, P, C, F, A, K = 'main_v1.2.R', 'src/mcmc_prepare.R', 'src/mcmc_conduction.R', 'src/mcmc_functions_v1.6.R', 'src/check_ages.R', 'src/prior_climate.R'
OLD_PREP, TRAFO = 'src/mcmc_prep_v1.2.R', 'src/all_trafo.R'
STATUS = ('gebunden', 'teilweise_gebunden', 'quellkonflikt', 'fehlend')
CONDITION_STATUS = ('belegt', 'bedingt', 'offen')
SOURCES = {M: '82486f99bafff08af2b51475ed225501d26cec1a0322a286b97a248ebd2f0520', P: 'b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e',
           C: '47c59c508e15150f7d6fda775a4f6bf6c14f0148e10cefd0014fa1ce6cfe5a9d', F: 'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c',
           A: 'c7f92e6030d83540c14f29a2e9d0ad07c74104dfe86befebda921c1cffbd2729', K: '77cfe4e2297abf5f0dd8f7179721702bb1a15e7f9989c9c62374a2fdda642ebd'}
EXTRA_READ = (OLD_PREP, TRAFO)
ACTIVE = (M, P, C, F, A, K)
OLD = 'in/core_data_all_andrea_taxa_additional_sites.rdata'
CSV_KINNERET = 'my_kinneret/data/in/core/pollen_counts_epd_taxanomy.csv'
BIOM = 'my_kinneret/data/in/core/core_data.rdata'
RHO = 'my_kinneret/data/out/age_depth/rho_depth_given_age.rdata'
GIESECKE = 'in/epd_giesecke_2016/quant_files/'
BINDING, REFS, MISSING, AUDIT, RUNTIME = ('kinneret_binding_2026_11_02/README.md', 'kinneret_references_2026_11_02/README.md', 'MISSING_PROJECT_FILES_2026_11_02.md',
                                          'climate_modelling_audit_2026_11_02/README.md', 'r_runtime_crosscheck_2026_11_03/README.md')


def r(file, line, start, note=None):
    return {'file': file, 'line': line, 'starts_with': start, **({'note_de': note} if note else {})}


def src(role, root, rel):
    return {'role': role, 'root': root, 'relative_path': rel}


def item(ident, group, label, status, why, reads, sources=(), bundles=(), affects=True, missing_ids=(), evidence=(), consumer=None):
    return {'id': ident, 'group': group, 'label_de': label, 'status': status, 'status_reason_de': why, 'code_reads': list(reads), 'diss_sources': list(sources),
            'phase1_bundles': list(bundles), 'affects_kinneret_result': affects, 'missing_list_ids': list(missing_ids), 'evidence': list(evidence), 'consumer_object': consumer}


LOAD_CORE = r(M, 107, 'load(paste0("in/core_data_additional_sites_v1.2.rdata"))')
INPUTS = [
    # the base entry of slot 3
    item('core_container', 'kern', 'Finaler Sammelcontainer der Zusatzorte', 'fehlend', 'Die Datei fehlt am Verbraucherpfad. Der ältere Sammelcontainer hat vier Standorte und eine andere Struktur.',
         [LOAD_CORE, r(M, 146, 'taxa_names <- core.data[[3]]$all.available.taxa$taxa.names'), r(M, 654, 'if( is.null(core.data[[station_id]]) | station_id == 469 ) next()')],
         [src('fehlt_am_verbraucherpfad', 'original', 'in/core_data_additional_sites_v1.2.rdata'), src('aelterer_kandidat_kein_ersatz', 'original', OLD)], ['final_base'], True, ['F01'], [BINDING, MISSING], 'core.data[[3]]'),
    item('site_name', 'kern', 'Standortname in Slot 3', 'quellkonflikt', 'Der ältere Container nennt „Kinneret“, alle Schalter prüfen „Lake Kinneret“. main_v1.2.R setzt den Namen für Slot 3 nicht. Der Wert des finalen Containers ist unbekannt.',
         [r(A, 7, 'sitename <- core.data[[station_id]]$sitename'), r(P, 790, 'if(sitename == "Lake Kinneret") which.site <- 105'), r(P, 881, 'if(length(human_impact) != 0 & !any(sitename == c("Lake Kinneret","En Gedi","Birkat Ram")) ){'),
          r(P, 1145, 'if(sitename == "Lake Kinneret")  sources$speleothem <- sources$marine_ice_core <- 0', 'nicht erreicht: case_studies ist FALSE'),
          r(P, 1209, 'if( !inside.giesecke & add.sites & lat <= 36 & !any(sitename == c("Dead Sea","Birkat Ram","Lake Kinneret")) ){'), r(P, 1248, 'if(any(sitename == c("Lake Kinneret", ""))){')],
         [src('aelterer_kandidat_kein_ersatz', 'original', OLD), src('fehlt_am_verbraucherpfad', 'original', 'in/core_data_additional_sites_v1.2.rdata')], ['final_base'], True, ['F01'], [BINDING], 'sitename'),
    item('taxa_spectrum', 'kern', 'Pollenspektrum (Proben × Taxa)', 'quellkonflikt', 'Der ältere Stand hat 160×12 und ist mit der CSV zahlengleich. Seine Spalten 5, 6 und 8 tragen aber andere Namen, als main_v1.2.R dort setzt. Anders als bei Prespa, En Gedi und Ohrid ersetzt der Code die Kinneret-Zahlen nicht aus der CSV.',
         [r(M, 153, 'colnames( core.data[[3]]$all.available.taxa$taxa.spectrum) <- taxa_names'), r(A, 14, 'all.avail <- core.data[[station_id]]$all.available.taxa$taxa.spectrum'), r(A, 15, 'all.avail[which(is.na(all.avail))] <- 0'),
          r(P, 83, 'all.avail[all.avail == 0] <- 1e-6'), r(P, 93, 'all.avail.age[,i] <-   rho.d.g.a %*% as.matrix(all.avail[,i])'), r(P, 1435, 'num_taxa <- ncol(all.avail)'), r(P, 1454, 'taxa.appear.core <- all.avail.age[,which_taxa]')],
         [src('aelterer_kandidat_kein_ersatz', 'original', OLD), src('zaehlwerte_csv', 'additional', CSV_KINNERET), src('anderer_biomzweig_kein_ersatz', 'additional', BIOM)], ['final_base'], True, ['F01'], [BINDING], 'all.avail'),
    item('taxon_names', 'kern', 'Taxonnamen und Reihenfolge', 'quellkonflikt', 'main_v1.2.R überschreibt die Positionen 5, 6 und 8. Im älteren Stand stehen dort Pinus cembra, Betula und Vitis sylvestris. Die finale Grundreihenfolge fehlt.',
         [r(M, 147, 'taxa_names[5] <- "Atriplex Halimus"', 'sofort durch Zeile 148 überschrieben'), r(M, 148, 'taxa_names[5] <- "Olea Europaea"'), r(M, 149, 'taxa_names[6] <- "Phillyrea Latifolia"'), r(M, 150, 'taxa_names[8] <- "Pistacia Lentiscus"'),
          r(M, 152, 'core.data[[3]]$all.available.taxa$taxa.names <- taxa_names'), r(P, 1452, 'taxa.sampled <- colnames(all.avail)[which_taxa]')],
         [src('aelterer_kandidat_kein_ersatz', 'original', OLD)], ['final_base', 'historical_tf'], True, ['F01'], [BINDING], 'taxa.sampled'),
    item('depth', 'kern', 'Tiefenvektor', 'teilweise_gebunden', '160 Tiefen stimmen in vier erhaltenen Quellen exakt überein. Der finale Container fehlt.',
         [r(A, 178, 'depth <- core.data[[station_id]]$depth'), r(A, 195, 'which_depths <- core.data[[station_id]]$depth %in% depth'), r(P, 14, 'rho.d.g.a[i,j] <- rho_depth_given_age[i,(depth[j]-min(depth, na.rm = T))+1]')],
         [src('aelterer_kandidat_kein_ersatz', 'original', OLD), src('zaehlwerte_csv', 'additional', CSV_KINNERET), src('chronologie_matrix', 'additional', RHO)], ['final_base', 'final_chronology'], True, ['F01'], [BINDING], 'depth'),
    item('age_grid', 'kern', 'Altersraster age[[2]]', 'quellkonflikt', 'Der Verbraucher erwartet eine Liste. bacon_kinneret.R schreibt zuerst eine Dreierliste, deren zweites Element das 50-Jahres-Raster ist, und überschreibt denselben Dateinamen später mit einem Mittelwertvektor.',
         [r(M, 656, 'age.reso <- 2'), r(A, 179, 'site.age <- age <- core.data[[station_id]]$age[[age.reso]]'), r(A, 223, 'depths.timeslice.id <- list( which(site.age <= 0)'), r(P, 728, 'which.modern.avail <- which(age <= 11000)')],
         [src('chronologie_matrix', 'additional', RHO), src('schreiber_liste_und_matrix', 'additional', 'my_kinneret/src/bacon_kinneret.R')], ['final_chronology'], True, ['F01'], [BINDING], 'age, site.age'),
    item('rho_depth_given_age', 'kern', 'Dichte Tiefe gegeben Alter, rho[[2]]', 'quellkonflikt', 'Erhalten ist eine 160×1772-Matrix. Der Verbraucher erwartet das zweite Listenelement, einen anderen Datentyp.',
         [r(A, 177, 'rho_depth_given_age <- core.data[[station_id]]$rho_depth_given_age[[age.reso]]'), r(P, 14, 'rho.d.g.a[i,j] <- rho_depth_given_age[i,(depth[j]-min(depth, na.rm = T))+1]'), r(P, 648, 'predictors$pollen <- quali.reconstr.ref.age <- (rho.d.g.a %*% as.matrix(quali.reconstr.ref.depth))[,1]')],
         [src('chronologie_matrix', 'additional', RHO), src('schreiber_liste_und_matrix', 'additional', 'my_kinneret/src/bacon_kinneret.R')], ['final_chronology'], True, ['F01'], [BINDING], 'rho.d.g.a'),
    item('coordinates', 'kern', 'Koordinaten lat/lon', 'teilweise_gebunden', 'Der ältere Container hat 32,79° N und 35,53° E. Der Wert des finalen Containers ist nicht belegt.',
         [r(K, 9, 'site.lat <- lat <- core.data[[station_id]]$lat'), r(K, 10, 'site.lon <- lon <- core.data[[station_id]]$lon'), r(P, 568, 'lat.id <- which.min(abs(giesecke.lat - site.lat))'), r(P, 755, 'distances[i] <- distance(site.lat,site.lon,iso.lat[i],iso.lon[i])'), r(P, 1209, 'if( !inside.giesecke & add.sites & lat <= 36')],
         [src('aelterer_kandidat_kein_ersatz', 'original', OLD)], ['final_base'], True, ['F01'], [BINDING], 'lat, lon'),
    item('country_elevation', 'kern', 'Land und Höhe', 'teilweise_gebunden', 'Der ältere Container hat Israel und −210. Beide Felder steuern im Slot-3-Pfad keinen Rechenwert, müssen aber vorhanden sein.',
         [r(P, 1036, 'if(core.data[[station_id]]$elevation >= 2000 && add.sites == FALSE) which.site <- 3'), r(P, 1080, 'if(!is.null(predictors$pollen) & core.data[[station_id]]$country != "Israel"){')],
         [src('aelterer_kandidat_kein_ersatz', 'original', OLD)], ['final_base'], False, ['F01'], [BINDING], 'country, elevation'),
    item('ap_reference', 'kern', 'AP-Referenz (tree_percentage)', 'gebunden', 'Die CSV liegt am passenden Pfad und wird vom Code selbst in Slot 3 geschrieben, unabhängig vom fehlenden Container. Die Alterskurve hängt zusätzlich an rho[[2]].',
         [r(M, 143, 'kinneret_core <- read.csv(file="../../reconstructions/my_reconst/my_kinneret/data/in/core/pollen_counts_epd_taxanomy.csv",header=T)'), r(M, 144, 'core.data[[3]]$qualitativ_data$qualitativ_reconstruction <- kinneret_core$tree_percentage[!is.na(kinneret_core$tree_percentage)]'),
          r(P, 639, 'quali.reconstr.ref.depth <- core.data[[station_id]]$qualitativ_data$qualitativ_reconstruction'), r(P, 648, 'predictors$pollen <- quali.reconstr.ref.age')],
         [src('verbraucherpfad', 'additional', CSV_KINNERET)], [], True, [], [REFS], 'predictors$pollen'),
    item('human_impact', 'kern', 'Fenster menschlicher Einfluss', 'gebunden', 'Das Literal c(-60,6500) steht in main_v1.2.R. Offen bleibt nur die Randwertregel von inside.range.',
         [r(M, 154, 'core.data[[3]]$human_impact <- c(-60,6500)'), r(P, 711, 'human_impact <- core.data[[station_id]]$human_impact'), r(P, 716, 'hi_ages_time <- !inside.range(pollen.age,human_impact)'), r(P, 878, 'human_impact <- core.data[[station_id]]$human_impact')], [], [], True, [], [REFS], 'human_impact'),
    # transfer functions
    item('tf_container', 'transferfunktionen', 'Numerische Transferfunktionen', 'fehlend', 'Die Datei fehlt. Nur die Schreibzeile in all_trafo.R ist erhalten. Die sieben modernen Artenmodelle sind kein Ersatz.',
         [r(M, 569, 'load("out/transfer_functions/mixed_partial_arts_emd.rdata")'), r(M, 577, 'tf.x1[i,] <- transfer.functions[[i]]$prod.x1'), r(M, 578, 'tf.x2[i,] <- transfer.functions[[i]]$prod.x2'), r(M, 588, 'tf.x1[i,] <- tf.x1[i,] / max(tf.x1[i,])'), r(M, 589, 'tf.x2[i,] <- tf.x2[i,] / max(tf.x2[i,])')],
         [src('fehlt_am_verbraucherpfad', 'original', 'out/transfer_functions/mixed_partial_arts_emd.rdata'), src('schreibzeile_all_trafo_529', 'original', TRAFO)], ['historical_tf'], True, ['F02'], [MISSING], 'transfer.functions, tf.x1, tf.x2'),
    item('tf_axes', 'transferfunktionen', 'Klimaachsen x1range.bck, x2range.bck', 'fehlend', 'Die Achsen stehen im fehlenden Container.',
         [r(M, 570, 'x1range.bck <- transfer.functions[[1]]$x1range.bck'), r(M, 571, 'x2range.bck <- transfer.functions[[1]]$x2range.bck'), r(P, 1445, 'x1range.bck <- transfer.functions[[1]]$x1range.bck'), r(P, 1446, 'x2range.bck <- transfer.functions[[1]]$x2range.bck'), r(P, 1934, 'tann_range <- seq( min(x1range.bck), max(x1range.bck), length.out = 1000)'), r(K, 159, 'tann_dens <- density(tann_temp[!is.na(tann_temp)], from = min(x1range.bck)')],
         [src('fehlt_am_verbraucherpfad', 'original', 'out/transfer_functions/mixed_partial_arts_emd.rdata')], ['historical_tf'], True, ['F02'], [MISSING], 'x1range.bck, x2range.bck'),
    item('taxa_name_pool', 'transferfunktionen', 'Namenspool taxa.distribution.ww$names', 'teilweise_gebunden', 'Eine Datei mit passendem Pfadsuffix liegt vor. Gleichheit mit dem verlorenen Automount-Original ist nicht belegt, und die Zuordnung Name zu TF-Index braucht den fehlenden Container. Die Kinneret-Kopie ist ein anderer Stand.',
         [r(M, 433, 'load("/automount/agh/Projects/s6tinetz/dissertation/palaeo_sfb/sciebo/diss_laptop/reconstructions/my_reconst/my_ohrid/data/in/climate_data/taxa_distribution_ww.rdata")'), r(M, 434, 'taxa.distribution.ww$dist <- taxa.distribution.ww$dist[,,-c(8,10,152,153)]'), r(M, 435, 'taxa.distribution.ww$names <- taxa.distribution.ww$names[-c(8,10,152,153)]'), r(M, 436, 'taxa.distribution.ww$names[110] <- "Juglans regia"'),
          r(P, 1460, 'taxa.merge <- anyNA(match(taxa.sampled,taxa.distribution.ww$names))'), r(P, 1573, 'which_taxa <- match(taxa.sampled,taxa.distribution.ww$names)')],
         [src('passendes_pfadsuffix', 'additional', 'my_ohrid/data/in/climate_data/taxa_distribution_ww.rdata'), src('anderer_stand_kein_ersatz', 'additional', 'my_kinneret/data/in/climate_data/taxa_distribution_ww.rdata')], ['historical_tf'], True, [], [MISSING, AUDIT], 'which_taxa'),
    item('tf_current', 'transferfunktionen', 'tf.x1.current, tf.x2.current (abgeleitet)', 'fehlend', 'Abgeleitet aus dem fehlenden Container. Der Nachbau hat dafür nur Kunstwerte.',
         [r(P, 1574, 'tf.x1.current <- tf.x1[which_taxa,]'), r(P, 1575, 'tf.x2.current <- tf.x2[which_taxa,]'), r(P, 1535, 'tf.x1.current <- tf.x2.current <- array(NA, dim = c(length(which_taxa), dims))', 'Zweig bei Mehrfachzuordnung'), r(P, 1581, 'tf.x1.current <- tf.x1.current/rowSums(tf.x1.current, na.rm=T)'), r(P, 1582, 'tf.x2.current <- tf.x2.current/rowSums(tf.x2.current, na.rm=T)'), r(P, 1631, 'tf.x1.current.mean[j] <- weightedMedian(x1range.bck,tf.x1.current[j,] )'), r(P, 1664, 'which.tf.missing <- which(is.na(tf.x1.current.mean))'), r(P, 1897, 'tf_tann_sample <- sample(x1range.bck,size=10000,prob = tf.x1.current[i,],replace = T)'), r(P, 1899, 'tf_pann_sample <- sample(x2range.bck,size=10000,prob = tf.x2.current[i,],replace = T)'), r(C, 146, 'posterior_temp <- reconstruction(posterior_temp=posterior_temp,tf.x1.current[posterior_temp$z,]')],
         [src('fehlt_am_verbraucherpfad', 'original', 'out/transfer_functions/mixed_partial_arts_emd.rdata')], ['historical_tf'], True, ['F02'], [MISSING], 'tf.x1.current, tf.x2.current'),
    # climate
    item('cru_pann_ww', 'klima', 'CRU-PANN global, Stationsmaske', 'fehlend', 'Die Datei fehlt. Regionale Ausschnitte sind kein Ersatz.',
         [r(M, 325, 'precip <-  nc_open("../../reconstructions/cru_nh/cru_pann_ww.nc", write=FALSE, readunlim=FALSE)'), r(M, 327, 'lon_ww <- ncvar_get(nc=precip,varid="lon")'), r(M, 329, 'p.ann.ww <- ncvar_get(nc=precip,varid="pre")'), r(K, 6, 'nh_mask <- ifelse(ncvar_get(nc=precip,varid="stn") >= 0,0,NA )'), r(K, 156, 'pann_temp <- p.ann.ww*kg_map_temp')], [src('fehlt_am_verbraucherpfad', 'historical', 'cru_nh/cru_pann_ww.nc')], ['cru_kgc_inputs'], True, ['F03'], [MISSING], 'p.ann.ww, nh_mask'),
    item('cru_tann_ww', 'klima', 'CRU-TANN global', 'fehlend', 'Die Datei fehlt. Regionale Ausschnitte sind kein Ersatz.',
         [r(M, 326, 'temp <-  nc_open("../../reconstructions/cru_nh/cru_tann_ww.nc", write=FALSE, readunlim=FALSE)'), r(M, 330, 't.ann.ww <- ncvar_get(nc=temp,varid="tmp")'), r(K, 154, 'tann_temp <- t.ann.ww*kg_map_temp')], [src('fehlt_am_verbraucherpfad', 'historical', 'cru_nh/cru_tann_ww.nc')], ['cru_kgc_inputs'], True, ['F04'], [MISSING], 't.ann.ww'),
    item('cru4_pann', 'klima', 'CRU4-PANN Zeitreihe 50:115', 'fehlend', 'Die Datei fehlt. Ohne Zeitachse ist 50:115 kein belegter Kalenderbezug.',
         [r(M, 343, 'precip_yearly <-  nc_open("../../reconstructions/cru_nh/cru4_pann_v1.2.nc", write=FALSE, readunlim=FALSE)'), r(M, 346, 'pann_v1.2 <- ncvar_get(nc=precip_yearly,varid="pre")[,,50:115]'), r(K, 71, 'pann_site <- pann_v1.2[loc_lon_idx,loc_lat_idx,]')], [src('fehlt_am_verbraucherpfad', 'historical', 'cru_nh/cru4_pann_v1.2.nc')], ['cru_kgc_inputs', 'anchor_variant'], True, ['F05'], [REFS], 'pann_site'),
    item('cru4_tann', 'klima', 'CRU4-TANN Zeitreihe 50:115', 'fehlend', 'Die Datei fehlt. Ohne Zeitachse ist 50:115 kein belegter Kalenderbezug.',
         [r(M, 344, 'temp_yearly <-  nc_open("../../reconstructions/cru_nh/cru4_tann_v1.2.nc", write=FALSE, readunlim=FALSE)'), r(M, 347, 'tann_v1.2 <- ncvar_get(nc=temp_yearly,varid="tmp")[,,50:115]'), r(K, 70, 'tann_site <- tann_v1.2[loc_lon_idx,loc_lat_idx,]')], [src('fehlt_am_verbraucherpfad', 'historical', 'cru_nh/cru4_tann_v1.2.nc')], ['cru_kgc_inputs', 'anchor_variant'], True, ['F06'], [REFS], 'tann_site'),
    item('climatezones', 'klima', 'Köppen-Geiger-Tabelle climatezones', 'fehlend', 'Der Code lädt climatezones nicht aus der Datei daneben: Sie enthält nur kgc.distribution.ww. In den R-Quellen wird der Name nirgends zugewiesen. Als Anbieter kommt das Paket kgc infrage; dessen Fassung ist ungebunden und wurde hier nicht geprüft.',
         [r(M, 360, 'cz <- climatezones'), r(K, 33, 'cz <- climatezones'), r(K, 39, 'site.info <- data.frame(site.info,ClimateZ=LookupCZ(site.info))'), r(K, 52, 'lon.cz <- cz$Lon[which(cz$Cls == site.kgcz[i])]')], [], ['cru_kgc_inputs'], True, [], [REFS], 'cz, site.kgcz'),
    # isotopes
    item('reference_speleo_iso', 'isotope', 'Workspace reference_speleo_iso', 'fehlend', 'Die Datei fehlt. Welche der Ladedateien d18o.age liefert, ist nicht belegt.', [r(M, 388, 'load("in/reference_speleo_iso.rdata")')], [src('fehlt_am_verbraucherpfad', 'original', 'in/reference_speleo_iso.rdata')], ['isotope_workspaces'], True, ['F07'], [MISSING], 'd18o.age?'),
    item('sisal_speleo_info', 'isotope', 'SISAL speleo_info.csv', 'teilweise_gebunden', 'Unter climate_modelling liegt ein Namenskandidat ohne Versionsbeleg. Im aktiven Pfad gibt es keinen Leser.', [r(M, 391, 'speleo_eu_info <- read.csv("../../stalage_speleodatabase/sisal/SISAL_V2/speleo_info.csv"')], [src('kandidat_version_offen', 'climate_modelling', 'speleothems/speleo_data/speleo_info.csv')], ['isotope_workspaces'], False, ['F14'], [AUDIT], 'speleo_eu_info'),
    item('sisal_ages_info', 'isotope', 'SISAL ages_info.csv', 'teilweise_gebunden', 'Unter climate_modelling liegt ein Namenskandidat ohne Versionsbeleg. Im aktiven Pfad gibt es keinen Leser.', [r(M, 392, 'ages_eu_info <- read.csv("../../stalage_speleodatabase/sisal/SISAL_V2/ages_info.csv"')], [src('kandidat_version_offen', 'climate_modelling', 'speleothems/speleo_data/ages_info.csv')], ['isotope_workspaces'], False, ['F15'], [AUDIT], 'ages_eu_info'),
    item('sisal_d18_d13', 'isotope', 'SISAL d18_d13.RData', 'fehlend', 'Die Datei fehlt. Welche der Ladedateien d18o.age liefert, ist nicht belegt.', [r(M, 394, 'load("../../stalage_speleodatabase/sisal/SISAL_V2/d18_d13.RData")')], [src('fehlt_am_verbraucherpfad', 'historical', 'stalage_speleodatabase/sisal/SISAL_V2/d18_d13.RData')], ['isotope_workspaces'], True, ['F08'], [MISSING], 'd18o.age?'),
    item('isotope_info', 'isotope', 'Workspace isotope_info', 'fehlend', 'Die Datei fehlt. Welche der Ladedateien d18o.age liefert, ist nicht belegt.', [r(M, 397, 'load("in/isotope_info.rdata")')], [src('fehlt_am_verbraucherpfad', 'original', 'in/isotope_info.rdata')], ['isotope_workspaces'], True, ['F09'], [MISSING], 'd18o.age?'),
    item('isotope_speleo_other', 'isotope', 'Workspace isotope_speleo_other', 'fehlend', 'Die Datei fehlt. Welche der Ladedateien d18o.age liefert, ist nicht belegt.', [r(M, 401, 'load("in/isotope_speleo_other.rdata")')], [src('fehlt_am_verbraucherpfad', 'original', 'in/isotope_speleo_other.rdata')], ['isotope_workspaces'], True, ['F10'], [MISSING], 'd18o.age?'),
    item('d18o_age', 'isotope', 'Objekt d18o.age', 'fehlend', 'Das Objekt wird in keiner R-Datei von local_reconstr_all_epd erzeugt, sondern nur aus einem der fehlenden Workspaces geladen. Koordinaten, Namen, Alter und Reihen fehlen.',
         [r(P, 751, 'iso.lat <- as.numeric(d18o.age$lat[1:170])'), r(P, 752, 'iso.lon <- as.numeric(d18o.age$lon[1:170])'), r(P, 766, 'mean[i] <- mean( diff( d18o.age$age[[speleo.check[i]]] ), na.rm=T )'), r(P, 767, 'included.ages[i] <- length( which(inside.range( site.age ,range(d18o.age$age[[speleo.check[i]]]) )) ) / length(site.age)'), r(P, 794, 'iso.ref <- d18o.age$splined.series[[which.site]]'), r(P, 828, 'iso.ref <- d18o.age$splined.series[[170]]'), r(P, 1038, 'iso.ref.time <- d18o.age$splined.series[[ (length(d18o.age$loc_nr)-3) + which.site ]]')], [], ['isotope_workspaces', 'reference_variant'], True, [], [REFS], 'd18o.age'),
    item('soreq_index', 'isotope', 'Index der Soreq-Reihe', 'quellkonflikt', 'Drei Quellstände nennen 37, 105 und 170. Ohne d18o.age sind sie keiner Reihe zuzuordnen.',
         [r(P, 790, 'if(sitename == "Lake Kinneret") which.site <- 105'), r(P, 828, 'iso.ref <- d18o.age$splined.series[[170]]'), r(OLD_PREP, 0, 'FOUND_BELOW')], [src('aeltere_vergleichsfassung', 'original', OLD_PREP)], ['reference_variant'], True, [], [REFS], 'iso.ref'),
    item('medstack_series', 'isotope', 'Marine Referenzreihe (MEDSTACK)', 'fehlend', 'Die Auswahlregel ist belegt, die Reihe selbst fehlt. Der Code wählt über die Länge von d18o.age$loc_nr, nicht über einen Namen.',
         [r(P, 1038, 'iso.ref.time <- d18o.age$splined.series[[ (length(d18o.age$loc_nr)-3) + which.site ]]'), r(P, 1039, 'if(which.site < 3 ) iso.ref.time$isotopes <- -iso.ref.time$isotopes'), r(P, 1044, 'predictors$iso_time <- iso.ref.time$isotopes')], [], ['isotope_workspaces', 'reference_variant'], True, [], [REFS], 'predictors$iso_time'),
    item('giesecke_grid', 'isotope', 'Giesecke-Gitter (nur Koordinaten)', 'gebunden', 'Die CSV-Dateien liegen am Verbraucherpfad. Im Slot-3-Pfad liest der Code nur Lat und Lon, die Inhaltsspalten nicht.',
         [r(M, 407, 'epd.giesecke.files <- list.files("in/epd_giesecke_2016/quant_files/")'), r(M, 412, 'all.taxa.giesecke[[i]] <- read.csv(paste0("in/epd_giesecke_2016/quant_files/",epd.giesecke.files[i]))'), r(P, 565, 'giesecke.lat <- all.taxa.giesecke[[1]]$Lat'), r(P, 566, 'giesecke.lon <- all.taxa.giesecke[[1]]$Lon')], [src('verbraucherpfad', 'original', GIESECKE + 'epdQuant_Abi.csv')], [], True, [], [], 'giesecke.lat, giesecke.lon'),
    # loaded or opened, without influence on the slot 3 result in the preserved flow
    item('insolation', 'nur_geladen', 'Insolationsreihe si.txt', 'fehlend', 'Die Datei fehlt. sol.irr hat außerhalb von main_v1.2.R:367–381 keinen aktiven Leser.', [r(M, 367, 'sol.irr <- read.table("../../insolation/si.txt")')], [src('fehlt_am_verbraucherpfad', 'historical', 'insolation/si.txt')], [], False, ['F13'], [MISSING], 'sol.irr'),
    item('station_ids', 'nur_geladen', 'station_ids_ref', 'gebunden', 'Die Datei liegt vor und enthält station_ids_ref. Das Objekt hat im aktiven Pfad keinen Leser.', [r(M, 431, 'load("in/station_ids_ref.rdata")')], [src('verbraucherpfad', 'original', 'in/station_ids_ref.rdata')], [], False, [], [MISSING], 'station_ids_ref'),
    item('kgc_rdata', 'nur_geladen', 'kgc_distribution_ww.rdata', 'gebunden', 'Die Datei liegt vor und enthält kgc.distribution.ww. Das Objekt hat im aktiven Pfad keinen Leser, die Tabelle climatezones steckt nicht darin.', [r(M, 359, 'load("in/kgc_distribution_ww.rdata")')], [src('verbraucherpfad', 'original', 'in/kgc_distribution_ww.rdata')], [], False, [], [MISSING], 'kgc.distribution.ww'),
    item('prob_c_kgc', 'nur_geladen', 'prob_c_kgc.rdata', 'fehlend', 'Die Datei fehlt. Ihre Objektnamen sind unbekannt, ein Einfluss ist daher weder belegt noch ausgeschlossen.', [r(M, 432, 'load("out/transfer_functions/prob_c_kgc.rdata")')], [src('fehlt_am_verbraucherpfad', 'original', 'out/transfer_functions/prob_c_kgc.rdata')], [], 'unbekannt', ['F11'], [MISSING], None),
    item('bond2001', 'nur_geladen', 'IRD-Reihe bond2001.csv', 'fehlend', 'Die Datei fehlt. ird_data wird im aktiven Pfad nur im nicht erreichbaren Zweig mcmc_prepare.R:872 gelesen.', [r(A, 238, 'ird_percent <- read.csv("../../other_isotopes/bond2001.csv")'), r(P, 872, 'if( any(sitename == c("","")) ){'), r(P, 873, 'df_speleo <- data.frame(age = ird_data$age, isotopes = ird_data$isotopes)')], [src('fehlt_am_verbraucherpfad', 'historical', 'other_isotopes/bond2001.csv')], [], False, ['F16'], [MISSING], 'ird_data'),
    item('pmip3_mask', 'nur_geladen', 'PMIP3-Eismaske', 'fehlend', 'Die Datei fehlt, wird aber geöffnet. Das Ergebnis der Prüfung in Zeile 462 wird in Zeile 468 unbedingt überschrieben.', [r(P, 451, 'glaciers_lgm <-  nc_open(paste0(in_path,"/pmip/lgm/glaciers_data/pmip3_21k_v0.nc"), write=FALSE, readunlim=FALSE)'), r(P, 462, 'if( length(which( c(hol_per,lg_per,eem_per) == 0)) > 1'), r(P, 468, 'stad_interstad <- FALSE')], [], [], False, ['F12'], [MISSING], 'glaciers'),
    # run time
    item('r_version', 'laufzeit', 'R-Fassung des Laufs', 'teilweise_gebunden', 'Die Eingabecontainer stammen aus R 3.5.0. Die Fassung des Laufs ist nicht belegt; die Quellen wurden bis 07.07.2023 geändert.', [r(M, 644, 'source("src/mcmc_functions_v1.6.R")', 'kein Leser im engeren Sinn: Laufzeitumgebung aller Zeilen')], [], ['runtime_rng'], True, [], [RUNTIME], None),
    item('pkg_laplacesdemon', 'laufzeit', 'rdirichlet, ddirichlet', 'teilweise_gebunden', 'Im Fenster 2018-04-23…2023-07-07 liefert LaplacesDemon beide Funktionen, in allen vier Fassungen textgleich. Die installierte Fassung ist unbekannt.', [r(M, 38, 'library(LaplacesDemon)'), r(F, 1248, 'new_weights <- rdirichlet(1,old_weights*posterior_temp$dirichlet_spread)[1,]'), r(F, 1269, 'weights_ratio <- ddirichlet(new_weights,rep(0.5,num_taxa)) / ddirichlet(old_weights,rep(0.5,num_taxa))')], [], ['runtime_rng'], True, [], [RUNTIME], None),
    item('pkg_matrixstats', 'laufzeit', 'weightedMedian', 'teilweise_gebunden', 'Im Fenster liefert allein matrixStats die Funktion. Die installierte Fassung ist unbekannt.', [r(M, 27, 'library(matrixStats)'), r(P, 1631, 'tf.x1.current.mean[j] <- weightedMedian(x1range.bck,tf.x1.current[j,] )')], [], ['runtime_rng'], True, [], [RUNTIME], None),
    item('pkg_rcppdist', 'laufzeit', 'r_truncnorm, d_truncnorm', 'teilweise_gebunden', 'Im Fenster gab es auf CRAN genau RcppDist 0.1.1. Die installierte Fassung ist unbekannt.', [r(F, 1566, 'out(i) = r_truncnorm(mean(i), sd(i), a(i), b(i));')], [], ['runtime_rng'], True, [], ['literal_rtruncnorm_binding_2026_11_09/README.md'], None),
    item('pkg_truncnorm', 'laufzeit', 'rtruncnorm aus dem Paket truncnorm', 'fehlend', 'Die Funktion setzt die Startwerte tf_sample_tann und tf_sample_pann und verbraucht Zufallszahlen. Das Paket ist nicht gebunden.', [r(P, 1886, 'library(truncnorm)'), r(P, 1957, 'posterior$tf_sample_tann[[1]][j,] <- rtruncnorm(1,'), r(P, 1958, 'posterior$tf_sample_pann[[1]][j,] <- rtruncnorm(1,')], [], ['runtime_rng'], True, [], [], None),
    item('pkg_spatstat_utils', 'laufzeit', 'inside.range', 'fehlend', 'Bestimmt die Masken und die Randwertregel bei 6500 a BP. Das Paket ist nicht gebunden.', [r(M, 50, 'library(spatstat.utils)'), r(P, 716, 'hi_ages_time <- !inside.range(pollen.age,human_impact)')], [], ['runtime_rng'], True, [], [REFS], None),
    item('pkg_rpref', 'laufzeit', 'psel (Pareto-Auswahl)', 'fehlend', 'Wird im Mehrkandidatenzweig der Höhlen gerufen; die Ausgabeordnung ist nicht gebunden.', [r(M, 49, 'library(rPref)'), r(P, 779, 'pareto.member <- psel(pareto.df, low(mean) * high(included.ages)* low(distance))')], [], ['runtime_rng'], True, [], [], None),
    item('pkg_biwavelet', 'laufzeit', 'wt und rcpp_wt_bases_morlet', 'fehlend', 'Nur im Speläothemzweig. rcpp_wt_bases_morlet ist in keiner R-Quelle definiert; der Anbieter ist nicht gebunden und wurde hier nicht geprüft.', [r(M, 52, 'library(biwavelet)'), r(P, 901, 'period <-  wt(df_speleo_nhp, s0 = s0, dj = dj, max.scale = max.scale, do.sig = FALSE)$period'), r(P, 934, 'wb[,a1] <- rcpp_wt_bases_morlet(k, scale[a1], -1)$daughter')], [], ['runtime_rng'], 'bedingt', [], [], None),
    item('pkg_signal', 'laufzeit', 'fftfilt', 'fehlend', 'Nur im Speläothemzweig über smoothing. Das Paket ist nicht gebunden.', [r(M, 24, 'library(signal)'), r(F, 2325, 'z <- fftfilt(rep(1, 8)/8, x_norm)'), r(P, 868, 'df_speleo <- smoothing(iso.ref)')], [], ['runtime_rng'], 'bedingt', [], [], None),
    item('pkg_mass', 'laufzeit', 'fitdistr', 'fehlend', 'Setzt die Anker und die Parameter der Transferfunktionen. Das Paket ist nicht gebunden.', [r(M, 62, 'library(MASS)'), r(K, 106, 'normparameter1 <- fitdistr(tann_site, "normal")'), r(P, 1916, 'fit_tf_tann_temp <- fitdistr(tf_tann_sample,densfun = "normal")')], [], ['runtime_rng'], True, [], [], None),
    item('pkg_kgc', 'laufzeit', 'LookupCZ, RoundCoordinates, CZUncertainty', 'fehlend', 'Bestimmen die Klimaklasse des Standorts. Das Paket ist nicht gebunden.', [r(M, 22, 'library(kgc)'), r(K, 37, 'site.info <- data.frame(site.info,rndCoord.lon = RoundCoordinates(site.info$Longitude)'), r(K, 40, 'site.info <- data.frame(site.info, CZUncertainty(site.info))')], [], ['cru_kgc_inputs', 'runtime_rng'], True, [], [REFS], None),
    item('pkg_armadillo', 'laufzeit', 'arma::randn in rmvnorm_rcpp', 'fehlend', 'Verbraucht beim Samplerstart Zufallszahlen. Paket und Compiler des Laufs sind nicht gebunden.', [r(M, 30, 'library(RcppArmadillo)'), r(F, 1720, 'arma::mat Y = arma::randn(n, ncols);'), r(C, 133, 'r_curr_all <- rmvnorm_rcpp(sample_length, rep(0,2*num_taxa*num_age), diag(2*num_taxa*num_age))')], [], ['runtime_rng'], True, [], [], None),
]
OTHER_SLOT_PATHS = [(112, 'my_prespa/data/in/core/core_data_counts.csv'), (121, 'my_gedi/data/in/core/en_gedi_counts.csv'), (127, 'my_gedi/data/out/age_depth/rho_depth_given_age.rdata'), (157, 'my_birkat_ram/data/in/core/pollen_counts_epd_taxanomy.csv'),
                    (161, 'my_iznik/data/in/core/core_information.csv'), (177, 'my_ohrid/data/in/core/core_data_counts.csv'), (186, 'my_gedi2/data/in/core/taxa_spectrum.csv'), (221, 'my_gedi2/data/in/age_depth/rho_depth_given_age_higher_resol.rdata'),
                    (231, 'my_birkat_ram/data/in/core/pollen_counts_composite.csv'), (256, 'my_birkat_ram/data/out/age_depth/rho_depth_given_age.rdata'), (265, 'my_van/data/in/core/taxa_spectrum.csv'), (288, 'my_van/data/in/age_depth/rho_depth_given_age.rdata'),
                    (294, 'my_van/data/in/core/overlap_check_2009.csv'), (295, 'my_van/data/in/core/overlap_check_2014.csv')]
SETTINGS = [('add.sites', 'TRUE', M, 73, 'add.sites <- T'), ('percent', 'FALSE', M, 80, 'percent <- F'), ('stations', '1:length(core.data) ohne 4', M, 615, 'if(add.sites ) stations <- c(1:length(core.data))[-4]'), ('gibbs', 'FALSE', M, 624, 'gibbs <- F'),
            ('sample_length', '25000', M, 630, 'sample_length <- 25000'), ('burn_in', '5000', M, 631, 'burn_in <- 5000'), ('thin_size', '5', M, 632, 'thin_size <- 5'), ('age.reso', '2', M, 656, 'age.reso <- 2'), ('sim_nr', '1', M, 657, 'sim_nr <- 1'),
            ('dims', '300', M, 658, 'dims <- 300'), ('null.ratio.check', 'FALSE', M, 659, 'null.ratio.check <- FALSE'), ('giesecke_prespa', 'FALSE', M, 669, 'giesecke_prespa <- F'), ('past_climate_module', 'FALSE', M, 670, 'past_climate_module <- F'), ('case_studies', 'FALSE', M, 672, 'case_studies <- F')]
RNG_CHAIN = [
    ('seed', P, 1297, 'set.seed(2022)', None), ('draw', P, 1298, 'prior_tann_sample <- rnorm(10000,prior.recent$mean.tmp,prior.recent$sd.tmp)', None), ('seed', P, 1301, 'set.seed(2022)', None), ('draw', P, 1302, 'prior_pann_sample <- rgamma(10000,shape=shape,rate=rate)', None),
    ('seed', P, 1332, 'set.seed(2022)', 'wenn multiple'), ('draw', P, 1333, 'beta_samples <- rbeta(10000,shape1,shape2)', 'wenn multiple'), ('seed', P, 1341, 'set.seed(2022)', 'wenn nicht multiple'), ('draw', P, 1342, 'beta_samples <- rbeta(10000,shape1,shape2)', 'wenn nicht multiple'),
    ('seed', P, 1354, 'set.seed(2022)', None), ('draw', P, 1355, 'beta_samples <- rbeta(10000,shape4,shape5)', None), ('draw', P, 1557, 'tann_merge <- tann_merge + sample(x=x1range.bck,1000', 'wenn Taxa zusammengeführt werden'), ('draw', P, 1558, 'pann_merge <- pann_merge + sample(x=x2range.bck,1000', 'wenn Taxa zusammengeführt werden'),
    ('draw', P, 1591, 'tann_tf_samples <- sample(x1range.bck,10000', None), ('draw', P, 1592, 'pann_tf_samples <- sample(x2range.bck,10000', None), ('draw', P, 1709, 'posterior$post_pollen[1] <- 0.01+rbeta(1,0.1,20)', 'wenn core oder giesecke'), ('draw', P, 1715, 'posterior$post_iso[1] <-  0.01+rbeta(1,0.1,20)', 'wenn marine_ice_core'), ('draw', P, 1716, 'posterior$post_iso_hi[1] <-  0.01+rbeta(1,0.1,20)', 'wenn marine_ice_core'),
    ('draw', P, 1723, 'posterior$post_spec[1] <- 0.01+rbeta(1,0.1,20)', 'wenn spectral'), ('draw', P, 1724, 'posterior$post_spec_hi[1] <- 0.01+rbeta(1,0.1,20)', 'wenn spectral'), ('draw', P, 1731, 'posterior$post_tann[1] <- prior.recent$mean.tmp - prior.recent$sd.tmp +rbeta(1,0.1,20)', 'wenn recent'), ('draw', P, 1734, 'posterior$post_pann[1] <- prior.recent$mean.pann + prior.recent$sd.pann +rbeta(1,0.1,20)', 'wenn recent'), ('draw', P, 1762, 'posterior$means_diff_tann[1] <- 1+rbeta(1,0.1,20)', None), ('draw', P, 1763, 'posterior$means_diff_pann[1] <- 0+rbeta(1,0.1,20)', None),
    ('seed', P, 1896, 'set.seed(2023)', 'je Taxon'), ('draw', P, 1897, 'tf_tann_sample <- sample(x1range.bck,size=10000', 'je Taxon'), ('seed', P, 1898, 'set.seed(2023)', 'je Taxon, letzter Reset vor dem Sampler'), ('draw', P, 1899, 'tf_pann_sample <- sample(x2range.bck,size=10000', 'je Taxon'),
    ('draw', P, 1957, 'posterior$tf_sample_tann[[1]][j,] <- rtruncnorm(1,', 'je Taxon'), ('draw', P, 1958, 'posterior$tf_sample_pann[[1]][j,] <- rtruncnorm(1,', 'je Taxon'), ('draw', P, 1979, 'post_tf_samples[,j-1] <- rnorm(10000,means[ct],sds[ct])', 'je Taxon'), ('draw', P, 1980, 'post_tf_samples[,j] <- rgamma(10000,shape=shapes[ct],rate=rates[ct])', 'je Taxon'),
    ('stop', P, 2057, 'stop()', 'aktiver Abbruch im erhaltenen Quellstand'), ('draw', C, 133, 'r_curr_all <- rmvnorm_rcpp(sample_length,', 'erste Ziehung in mcmc_conduction.R'),
]
BUNDLE_NOTES = {'posterior_selection': 'Ausgabeseite, keine Eingabe des Samplers', 'postprocessing_variant': 'Ausgabeseite, keine Eingabe des Samplers', 'numeric_original_output': 'Ausgabeseite, keine Eingabe des Samplers'}


def source_flags(site_name, core, speleo, inside_giesecke=False, lat=32.79):
    """Symbolic trace of mcmc_prepare.R:1189-1203, 1208-1214 and 1248-1252. The order matters: which_sources is read before line 1212 changes sources[4]."""
    sources = {1: 0, 2: core, 3: speleo, 4: 1}
    which = [i for i, value in sources.items() if value == 1]
    multiple = (len(which) == 2 and which in ([1, 4], [2, 4])) or (len(which) == 3 and which in ([1, 3, 4], [2, 3, 4]))
    spectral = 3 in which
    tann, marine = True, 1
    if not inside_giesecke and lat <= 36 and site_name not in ('Dead Sea', 'Birkat Ram', 'Lake Kinneret'):
        tann, multiple, marine = False, False, 0
    if site_name in ('Lake Kinneret', ''):
        tann = False
    return {'site_name': site_name, 'sources_core': core, 'sources_speleothem': speleo, 'marine_ice_core_after_1212': marine, 'active_source_indices_at_1189': which, 'multiple': multiple, 'spectral': spectral, 'tann': tann, 'wt_power_reachable': spectral}


def load_sources(base):
    lines = {}
    for rel, expected in SOURCES.items():
        raw = (base / rel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f'{rel} differs from the frozen binding')
        lines[rel] = raw.decode('utf-8', 'replace').split('\n')
    for rel in EXTRA_READ:
        lines[rel] = (base / rel).read_bytes().decode('utf-8', 'replace').split('\n')
    return lines


def check_ref(lines, ref):
    text = lines[ref['file']][ref['line'] - 1].strip()
    if not text.startswith(ref['starts_with']):
        raise ValueError(f"{ref['file']}:{ref['line']} differs: {text!r} does not start with {ref['starts_with']!r}")


def code(line):
    return line.split('#', 1)[0]


def describe(roots, source):
    # A historical ../../ path has no fixed place in today's diss layout, so both levels are probed; no directory is searched.
    tries = [roots[source['root']] / source['relative_path']] + ([roots['original'].parent.parent / source['relative_path']] if source['root'] == 'historical' else [])
    path = next((t for t in tries if t.is_file()), tries[0])
    entry = {**source, 'diss_path': Path(roots['diss_prefix'][source['root']], source['relative_path']).as_posix(), 'exists': path.is_file()}
    if path.is_file():
        entry.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return entry


def older_entry(base):
    data, reader = read_metadata(base / OLD)
    sites = children(children(data.object)['core.data'])
    fields = children(sites['3'])
    scalar = lambda name: np.asarray(small_value(fields[name], reader)).ravel()[0]
    return {'slot_R_1_based': 3, 'sitename': str(scalar('sitename')), 'lat': float(scalar('lat')), 'lon': float(scalar('lon')), 'country': str(scalar('country')), 'elevation': float(scalar('elevation')),
            'has_field_qualitativ_data': 'qualitativ_data' in fields, 'has_field_human_impact': 'human_impact' in fields, 'all_sitenames': [str(np.asarray(small_value(children(o)['sitename'], reader)).ravel()[0]) for o in sites.values()]}


def objects_in(path):
    data, _ = read_metadata(path)
    return sorted(children(data.object))


def evaluate_inside_giesecke(base, older):
    folder = base / GIESECKE
    listed = sorted(p.name for p in folder.iterdir() if not p.name.startswith('.'))
    rows = list(csv.DictReader((folder / listed[0]).open(encoding='utf-8-sig', newline='')))
    lat, lon = [float(row['Lat']) for row in rows], [float(row['Lon']) for row in rows]
    nearest = min(range(len(lat)), key=lambda i: abs(lat[i] - older['lat']))
    band = [lon[i] for i, value in enumerate(lat) if value == lat[nearest]]
    return {'scope': 'only mcmc_prepare.R:568-572 on the columns Lat and Lon, with the older coordinates; not a model step', 'first_listed_file': listed[0], 'files_listed': len(listed), 'last_listed_file': listed[-1], 'grid_cells': len(rows),
            'lat_range': [min(lat), max(lat)], 'lon_range': [min(lon), max(lon)], 'nearest_lat': lat[nearest], 'cells_on_nearest_lat': len(band), 'lon_range_on_nearest_lat': [min(band), max(band)], 'site_lat_used': older['lat'], 'site_lon_used': older['lon'],
            'inside_giesecke_space': bool(older['lon'] > min(band) - 1 and older['lon'] < max(band) + 1)}


def writers(base, pattern):
    hits, scanned, seen = [], 0, set()
    for path in base.rglob('*.R'):
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        scanned += 1
        for number, text in enumerate(path.read_bytes().decode('latin-1').split('\n'), 1):
            if re.search(pattern, code(text)):
                hits.append(f'{path.relative_to(base).as_posix()}:{number}')
    return sorted(hits), scanned


def conditions(giesecke, older):
    inside = giesecke['inside_giesecke_space']
    c = lambda ident, status, label, why, reads, depends, **extra: {'id': ident, 'status': status, 'label_de': label, 'reason_de': why, 'code_reads': reads, 'depends_on_inputs': depends, **extra}
    return [
        c('sources_giesecke', 'belegt', 'sources$giesecke bleibt 0', 'Der Zweig 591 verlangt !add.sites oder giesecke_prespa. Beides ist im gebundenen Quellstand falsch.', [r(P, 555, 'sources <- data.frame(giesecke = 0, core = 0, speleothem = 0, marine_ice_core = 0)'), r(P, 591, 'if( inside.giesecke & !add.sites | giesecke_prespa ){')], []),
        c('inside_giesecke', 'bedingt', 'inside.giesecke ist FALSE', 'Das Giesecke-Gitter endet im Süden bei 34,75° N mit einer einzigen Zelle bei 24,75° E. Mit den älteren Koordinaten ist die Raumbedingung falsch. Der finale Containerwert ist nicht belegt.', [r(P, 568, 'lat.id <- which.min(abs(giesecke.lat - site.lat))'), r(P, 572, 'inside.giesecke.space <- ( site.lon > (min(giesecke.site.lon)-1) & site.lon < (max(giesecke.site.lon)+1) )'), r(P, 575, 'inside.giesecke.time <- any(site.age <= 13000)'), r(P, 587, 'inside.giesecke <- inside.giesecke.space & inside.giesecke.time')], ['coordinates', 'giesecke_grid'], value_under_older_coordinates=bool(inside), evaluation=giesecke),
        c('sources_core', 'bedingt', 'sources$core', 'Der Zweig 636 wird erreicht, solange inside.giesecke falsch ist; dann wird sources$core gesetzt. Offen bleibt der Varianztest in Zeile 662, der die Alterskurve braucht.', [r(P, 636, '} else if( !inside.giesecke &  add.sites | sitename %in% c("Lake Prespa", "Lake Ohrid", "Lake Van") ){'), r(P, 661, 'sources$core <- 1'), r(P, 662, 'if( var(predictors$pollen[pollen.age %in% site.age], na.rm=T)== 0 ) sources$core <- 0')], ['coordinates', 'giesecke_grid', 'ap_reference', 'age_grid', 'rho_depth_given_age'], branch_reached_under_older_coordinates=not inside, variance_test_open=True),
        c('sources_speleothem', 'offen', 'sources$speleothem', 'Hängt an den Höhlenkoordinaten und Altersachsen in d18o.age. Das Objekt fehlt. Der Dissertationstext nennt Soreq-WPS für Kinneret; das bindet den Code nicht.', [r(P, 759, 'speleo.check <- which( distances <= 150 )'), r(P, 772, 'sources$speleothem <- 1'), r(P, 773, 'if(max(included.ages) <= 0.5) sources$speleothem <- 0'), r(P, 813, 'sources$speleothem <- 1'), r(P, 814, 'if(included.ages <= 0.5) sources$speleothem <- 0')], ['d18o_age', 'age_grid', 'coordinates', 'pkg_spatstat_utils']),
        c('sources_marine', 'bedingt', 'sources$marine_ice_core', 'Zeile 1072 setzt den Wert ohne Bedingung auf 1. Zeile 1212 setzt ihn für Standorte außerhalb der Namensliste wieder auf 0. Das hängt am Standortnamen.', [r(P, 1072, 'sources$marine_ice_core <- 1'), r(P, 1212, 'sources[4] <- 0')], ['site_name', 'coordinates', 'giesecke_grid']),
        c('multiple_spectral', 'offen', 'multiple und spectral', 'Folgen aus den vier Quellenschaltern. Die Tabelle source_flag_table führt alle Fälle auf.', [r(P, 1189, 'which_sources <- which(sources == 1)'), r(P, 1195, 'if( all(which_sources == c(2,4)) )  multiple <- TRUE'), r(P, 1200, 'if( all(which_sources == c(2,3,4)) )  multiple <- TRUE'), r(P, 1203, 'if( any(which_sources == 3) )  spectral <- TRUE')], ['site_name', 'd18o_age']),
        c('wt_power_reachable', 'offen', 'wt_power erreichbar', 'Läuft genau dann, wenn sources$speleothem 1 ist: in der Vorbereitung und in jeder Iteration über spectral. Das bleibt offen, solange d18o.age fehlt.', [r(P, 947, 'test <- c(wt_power(df_speleo_nhp$iso,length(df_speleo_nhp$iso), n.obs=n.obs, zeros=zeros, invflen=invflen)[coi_array == 1] )'), r(P, 966, 'test <- c(wt_power(df_speleo_nhp$iso,length(df_speleo_nhp$iso), n.obs=n.obs, zeros=zeros, invflen=invflen)[coi_array == 1] )'), r(F, 2021, 'if(spectral){'), r(F, 2023, 'predictant_spec <- c(wt_power(predictant_current[ages_spec][hi_ages_spec],length_ages_spec2, n.obs=n.obs, zeros=zeros, invflen=invflen)[coi_array == 1])')], ['d18o_age']),
        c('site_age_override', 'bedingt', 'site.age wird im Speläothemzweig ersetzt', 'Ist sources$speleothem 1, ersetzt Zeile 825–826 site.age durch 418 Werte mit einer Lücke von 2000 bis 14500 a BP, und Zeile 828 setzt Reihe 170. Beides passt nicht zu einem 0–9-ka-Kern und ist ohne Laufbindung nicht als Kinneret-Lauf zu lesen.', [r(P, 825, 'site.age <- seq(0,220500,by=500)'), r(P, 826, 'site.age <- site.age[-c(6:29)]'), r(P, 828, 'iso.ref <- d18o.age$splined.series[[170]]')], ['d18o_age', 'soreq_index']),
        c('recent', 'bedingt', 'recent', 'recent ist genau dann FALSE, wenn kein Alter ≤ 11000 vorkommt. Die 160 älteren Mittelalter reichen von −60 bis 9015. Das finale Raster age[[2]] fehlt.', [r(P, 728, 'which.modern.avail <- which(age <= 11000)'), r(P, 1183, 'recent <- TRUE'), r(P, 1184, 'if( length(which.modern.avail) == 0) recent <- FALSE'), r(P, 1264, 'if( recent & min(age) >= 2000 ){'), r(P, 1730, 'if(recent){'), r(C, 146, 'posterior_temp <- reconstruction('), r(F, 1844, 'if(recent){')], ['age_grid']),
        c('anchor_branch', 'bedingt', 'Zweig des Klimaankers', 'Bei mindestens einem Alter ≤ 0 nimmt der Code die lokale CRU4-Zeitreihe, sonst die Klassenmittel. Das ältere Altersraster hat Werte ≤ 0, das finale fehlt. Das ist eine andere Bedingung als recent.', [r(A, 223, 'depths.timeslice.id <- list( which(site.age <= 0)'), r(K, 69, 'if(length(depths.timeslice.id[[1]]) > 0){'), r(K, 109, 't.ann.kg <- t.ann.ww[lon.kg.prior,lat.kg.prior][which(cz.map[lon.kg.prior,lat.kg.prior] == 1)]')], ['age_grid', 'cru4_pann', 'cru4_tann', 'cru_pann_ww', 'cru_tann_ww', 'climatezones']),
        c('tann_flag', 'bedingt', 'tann', 'Bei „Lake Kinneret“ setzt Zeile 1249 FALSE. Bei einem anderen Namen setzt mit den älteren Koordinaten schon Zeile 1210 FALSE, schaltet dann aber auch multiple und die marine Reihe ab.', [r(P, 1208, 'tann <- TRUE'), r(P, 1209, 'if( !inside.giesecke & add.sites & lat <= 36'), r(P, 1210, 'tann <- FALSE'), r(P, 1249, 'tann <- FALSE'), r(F, 1939, 'if(tann){')], ['site_name', 'coordinates', 'giesecke_grid']),
        c('hi_time', 'belegt', 'hi_time ist FALSE', 'Im Zusatzorte-Zweig setzt Zeile 718 hi_time zurück. Die Altersmaske hi_ages_time bleibt bestehen.', [r(P, 713, 'hi_time <- FALSE'), r(P, 715, 'hi_time <- TRUE'), r(P, 718, 'if(add.sites) hi_time <- FALSE')], ['human_impact']),
        c('hi_spec', 'bedingt', 'hi_spec', 'Zeile 881 schließt „Lake Kinneret“ aus. Bei einem anderen Namen würde hi_spec TRUE. Der Zweig läuft nur bei sources$speleothem 1.', [r(P, 879, 'hi_spec <- FALSE'), r(P, 881, 'if(length(human_impact) != 0 & !any(sitename == c("Lake Kinneret","En Gedi","Birkat Ram")) ){'), r(P, 882, 'hi_spec <- TRUE')], ['site_name', 'd18o_age']),
        c('stad_interstad', 'belegt', 'stad_interstad ist FALSE', 'Zeile 468 setzt den Wert unbedingt. Die PMIP-Maske beeinflusst das Ergebnis deshalb nicht.', [r(P, 468, 'stad_interstad <- FALSE')], ['pmip3_mask']),
        c('taxa_merge', 'offen', 'Mehrfachzuordnung von Taxa', 'Ob ein Name fehlt, hängt an der finalen Taxonliste. Der Zweig zieht ohne eigenen Seed (1557–1558) und überschreibt alle Rechenvarianten mit der letzten.', [r(P, 1460, 'taxa.merge <- anyNA(match(taxa.sampled,taxa.distribution.ww$names))'), r(P, 1470, 'if( taxa.merge){'), r(P, 1557, 'tann_merge <- tann_merge + sample(x=x1range.bck,1000'), r(P, 1567, 'tf.x1.current[p,] <- tann_merge_dens$y/sum(tann_merge_dens$y)')], ['taxon_names', 'taxa_name_pool', 'tf_container']),
        c('tf_missing', 'offen', 'Fehlende Transferfunktionen', 'Welche Taxa ohne Funktion sind, hängt am fehlenden Container.', [r(P, 1664, 'which.tf.missing <- which(is.na(tf.x1.current.mean))'), r(P, 1678, 'num_taxa <- nrow(tf.x1.current)')], ['tf_container', 'taxon_names']),
        c('active_stop', 'belegt', 'Aktiver Abbruch vor dem Sampler', 'mcmc_prepare.R endet mit einem aktiven stop(). Im erhaltenen Quellstand wird mcmc_conduction.R daher nicht erreicht. Für den historischen Lauf muss eine andere Fassung gegolten haben; welche, ist nicht belegt.', [r(P, 2057, 'stop()'), r(M, 675, 'source("src/mcmc_prepare.R")'), r(M, 676, 'source("src/mcmc_conduction.R")')], []),
        c('rng_start', 'offen', 'Zufallszustand beim Samplerstart', 'Belegt ist nur die Kette: letzter Reset set.seed(2023) in Zeile 1898, danach weitere Ziehungen. Die Startwerte in 1709–1763 liegen schon davor, nach set.seed(2022) in Zeile 1354. Der Zustand hängt an RNGkind, R-Fassung, Paketen und allen fehlenden Eingaben.', [r(P, 1354, 'set.seed(2022)'), r(P, 1898, 'set.seed(2023)'), r(C, 133, 'r_curr_all <- rmvnorm_rcpp(sample_length,')], ['r_version', 'pkg_truncnorm', 'pkg_armadillo', 'tf_current', 'taxon_names', 'age_grid'], rngkind_or_rngversion_calls_in_active_sources=0),
    ]


def main():
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    base, extra, climate = Path(cfg['original_data_root']), Path(cfg['additional_source_root']), Path(cfg['climate_modelling_source_root'])
    roots = {'original': base, 'additional': extra, 'climate_modelling': climate, 'historical': base.parent,
             'diss_prefix': {'original': 'src/local_reconstr_all_epd', 'additional': 'src/my_reconst', 'climate_modelling': 'src/climate_modelling', 'historical': 'src'}}
    lines = load_sources(base)
    old_index = [i + 1 for i, text in enumerate(lines[OLD_PREP]) if re.search(r'sitename == "Lake Kinneret"\) which\.site <- 37', code(text))]
    if len(old_index) != 1:
        raise ValueError(f'expected one index 37 assignment, found {old_index}')
    for entry in INPUTS:
        for ref in entry['code_reads']:
            if ref['starts_with'] == 'FOUND_BELOW':
                ref.update(line=old_index[0], starts_with='if(sitename == "Lake Kinneret") which.site <- 37', note_de='ältere Vergleichsfassung, nicht aktiv geladen')
            check_ref(lines, ref)
        entry['diss_sources'] = [describe(roots, s) for s in entry['diss_sources']]
    trafo_line = lines[TRAFO][528].strip()
    if not trafo_line.startswith('save(transfer.functions, file = "out/transfer_functions/mixed_partial_arts_emd.rdata")'):
        raise ValueError('all_trafo.R:529 differs')
    older = older_entry(base)
    giesecke = evaluate_inside_giesecke(base, older)
    conds = conditions(giesecke, older)
    for entry in conds:
        for ref in entry['code_reads']:
            check_ref(lines, ref)
    ids = {e['id'] for e in INPUTS}
    for entry in conds:
        unknown = set(entry['depends_on_inputs']) - ids
        if unknown:
            raise ValueError(f"{entry['id']} names unknown inputs {unknown}")
    other_slots = []
    for number, rel in OTHER_SLOT_PATHS:
        if not any(rel.rsplit('/', 1)[-1] in text for text in [lines[M][number - 1]]):
            raise ValueError(f'main_v1.2.R:{number} does not name {rel}')
        other_slots.append({'line': number, 'relative_path': rel, 'diss_path': f'src/my_reconst/{rel}', 'exists': (extra / rel).is_file()})
    settings = []
    for name, value, file, number, start in SETTINGS:
        check_ref(lines, r(file, number, start))
        settings.append({'name': name, 'value': value, 'file': file, 'line': number, 'starts_with': start})
    chain = []
    for kind, file, number, start, condition in RNG_CHAIN:
        check_ref(lines, r(file, number, start))
        chain.append({'kind': kind, 'file': file, 'line': number, 'starts_with': start, 'condition_de': condition})
    spec = json.loads((ROOT / 'docs/thesis_research/kinneret_phase1_spec_2026_11_02/spec.json').read_text(encoding='utf-8'))
    bundles = [{'id': q['id'], 'label_de': q['label_de'], 'phase1_status': q['unresolved_status'], 'closed': False, 'inputs': [e['id'] for e in INPUTS if q['id'] in e['phase1_bundles']], **({'note_de': BUNDLE_NOTES[q['id']]} if q['id'] in BUNDLE_NOTES else {})} for q in spec['requirements']]
    flag_table = [source_flags(name, core, speleo) for name in ('Lake Kinneret', 'Kinneret') for core in (1, 0) for speleo in (1, 0)]
    d18o_writers, scanned = writers(base, r'd18o\.age\s*(<-|=[^=])|save\([^)]*d18o\.age')
    cz_writers, _ = writers(base, r'\bclimatezones\s*(<-|=[^=])|save\([^)]*climatezones')
    active_rng = {rel: [i + 1 for i, text in enumerate(lines[rel]) if re.search(r'\b(RNGkind|RNGversion)\s*\(', code(text))] for rel in ACTIVE}
    seeds = {rel: [i + 1 for i, text in enumerate(lines[rel]) if re.search(r'\bset\.seed\s*\(', code(text))] for rel in ACTIVE}
    counts = {status: sum(1 for e in INPUTS if e['status'] == status) for status in STATUS}
    report = {
        'kind': 'input_contract_of_the_kinneret_case_no_calculation_no_draw', 'date': '2026-11-09', 'case': {'site': 'Kinneret', 'slot_R_1_based': 3, 'branch': 'add.sites <- T'},
        'scope': 'Which inputs the preserved code reads for slot 3 and how far each is bound. Builds on the Phase-1 contract and does not replace it. Bound means present at the consumer path or written literally in the code; it never means bound to the historical run.',
        'status_values': {'gebunden': 'Datei am Verbraucherpfad vorhanden oder Literal im Code', 'teilweise_gebunden': 'Teil des Inhalts belegt oder Kandidat mit benannter offener Differenz', 'quellkonflikt': 'erhaltene Quellen widersprechen sich', 'fehlend': 'keine Quelle in diss am Verbraucherpfad oder keine Bindung vorhanden'},
        'sources_sha256': SOURCES, 'status_counts': counts, 'inputs': INPUTS, 'settings_literal': settings, 'other_slot_paths_not_affecting_kinneret': other_slots, 'older_container_slot_3': older, 'conditions': conds, 'source_flag_table': flag_table,
        'rng_chain': chain, 'rng_facts': {'set_seed_uncommented': seeds, 'rngkind_or_rngversion_uncommented': active_rng, 'seed_variable_seeds_in_main_line_65_has_no_active_reader': True, 'main_line_1055_is_after_the_station_loop': True, 'station_loop_lines': [651, 684]},
        'writer_search': {'scope': 'R files below local_reconstr_all_epd only, not a search of all of diss', 'files_scanned': scanned, 'd18o_age_assignments_or_saves': d18o_writers, 'climatezones_assignments_or_saves': cz_writers},
        'container_objects': {'in/kgc_distribution_ww.rdata': objects_in(base / 'in/kgc_distribution_ww.rdata'), 'in/station_ids_ref.rdata': objects_in(base / 'in/station_ids_ref.rdata')},
        'phase1_bundles': bundles, 'phase1_spec_sha256': hashlib.sha256((ROOT / 'docs/thesis_research/kinneret_phase1_spec_2026_11_02/spec.json').read_bytes()).hexdigest(), 'bundles_closed': 0, 'selected_variant': None,
        'draws_run': 0, 'sampler_run': False, 'r_scripts_run': False, 'models_trained': False, 'historical_reproduction_ready': False}
    (OUT / 'input_contract.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print(counts, len(INPUTS), len(conds), 'giesecke', giesecke['inside_giesecke_space'], 'writers', d18o_writers, cz_writers, 'scanned', scanned)
    print(older)
    print('other slots missing', [o['relative_path'] for o in other_slots if not o['exists']])
    print([(e['id'], s['relative_path'], s['exists']) for e in INPUTS for s in e['diss_sources'] if s['role'] != 'fehlt_am_verbraucherpfad' and not s['exists']])
    print([(e['id'], s['relative_path']) for e in INPUTS for s in e['diss_sources'] if s['role'] == 'fehlt_am_verbraucherpfad' and s['exists']])


if __name__ == '__main__':
    main()
