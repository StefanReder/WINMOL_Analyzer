xhost +
docker run --rm -it --name qgis --net host \
    -v /tmp/.X11-unix:/tmp/.X11-unix  \
    -v ./:/root/.local/share/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyser  \
    -e DISPLAY=unix$DISPLAY \
    --privileged \
    qgis/qgis:final-3_28_13 \
    qgis
