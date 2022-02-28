# from pyspark import SparkContext
# sc = SparkContext("local", "hdfs connector")

# client_url = "hdfs://192.168.54.153:9000/"
# URI = sc._gateway.jvm.java.net.URI
# Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
# FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
# fs = FileSystem.get(URI(client_url), sc._jsc.hadoopConfiguration())
# fileList = fs.listStatus(Path('/'))
# fileList = [x.toString() for x in fileList]
