FROM condaforge/miniforge3:4.10.1-0

WORKDIR /app

# Create the environment:
COPY environment.yml .
COPY tester.py .
COPY /bciavm .
RUN conda env create -f environment.yml
RUN echo "source activate bcienv" > ~/.bashrc
ENV PATH /opt/conda/envs/bcienv/bin:$PATH

CMD ["tester.py"]
ENTRYPOINT ["python"]

