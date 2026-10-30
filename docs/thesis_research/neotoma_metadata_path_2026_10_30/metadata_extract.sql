-- Read-only metadata request for Neotoma dataset 47525.
-- Prepared against the pinned public schema, NOT run on Neotoma data.
-- Preserve missing samples and inconsistent collection-unit links for review.
-- No climate values, age transfer, EPD identity, or MON16 depth is inferred.
SELECT
    ds.datasetid,
    ds.datasetname,
    ds.notes AS dataset_notes,
    ds.collectionunitid AS dataset_collectionunitid,
    cu.handle AS collectionunit_handle,
    cu.collunitname AS collectionunit_name,
    cu.location AS collectionunit_location,
    cu.notes AS collectionunit_notes,
    cu.siteid,
    st.sitename,
    smp.sampleid,
    smp.samplename,
    smp.labnumber,
    smp.notes AS sample_notes,
    smp.analysisunitid AS sample_analysisunitid,
    anu.analysisunitid AS resolved_analysisunitid,
    anu.collectionunitid AS analysisunit_collectionunitid,
    anu.analysisunitname,
    anu.depth AS analysisunit_depth_cm,
    anu.thickness AS analysisunit_thickness_cm,
    anu.igsn,
    anu.notes AS analysisunit_notes,
    CASE
        WHEN smp.sampleid IS NULL THEN 'no_samples'
        WHEN anu.analysisunitid IS NULL THEN 'missing_analysisunit'
        WHEN anu.collectionunitid <> ds.collectionunitid THEN 'collectionunit_mismatch'
        ELSE 'joined'
    END AS relation_status
FROM ndb.datasets AS ds
LEFT JOIN ndb.collectionunits AS cu ON cu.collectionunitid = ds.collectionunitid
LEFT JOIN ndb.sites AS st ON st.siteid = cu.siteid
LEFT JOIN ndb.samples AS smp ON smp.datasetid = ds.datasetid
LEFT JOIN ndb.analysisunits AS anu ON anu.analysisunitid = smp.analysisunitid
WHERE ds.datasetid = 47525
ORDER BY smp.sampleid;
