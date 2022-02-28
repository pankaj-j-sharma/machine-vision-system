import { Component, ElementRef, HostListener, OnInit, ViewChild } from '@angular/core';
import { Chart } from 'chart.js';
import { RestApiService } from 'src/app/services/restAPI/rest-api.service';

declare var jQuery: any;

export interface uploadFile {
  no: number
  file: File
  name: string
  path: string
  updated: number
  class: string
  tag: string
}

export interface PerfCharts {
  Id: string
  Name: string
  Data: number[]
  Labels: string[]
  BgColor: string[]
}

export interface Project {
  Id: string
  Name: string
  Description: string
  Resource: Number
  Type: Number
  Export: Number
  UpdatedOn: string
  Thumbnail: string
}

export interface ProjectData {
  Training: ImageTag[]
  Performance: Iteration[]
  Prediction: ImagePrediction[]
}

export interface Iteration {
  IterId: string
  Domain: string
  Type: string
  Precision: number
  Recall: number
  AbsPrecision: number
}

export interface ImageTag {
  Tag: string
  ImgUrl: string
  Title: string
}

export interface ImagePrediction {
  ImgUrl: string
  Result: string
}



@Component({
  selector: 'app-project',
  templateUrl: './project.component.html',
  styleUrls: ['./project.component.scss']
})
export class ProjectComponent implements OnInit {

  @ViewChild('nextStep') nextStep!: ElementRef;

  @ViewChild('elementToCheck') set performanceTab(performanceTab: any) {
    // here you get access only when element is rendered (or destroyed)
    if (this.activeStep == 2) {
      // this.performanceCharts.forEach(s=>this.generateCharts(s));
    }
  }

  exportOptions: any[] = [];
  projectTypes: any[] = [];
  allResources: any[] = [];
  projectCreated: boolean = false;
  projectSelected: boolean = false;
  dragAreaClass: string = "image-upload-wrap";
  fileList: uploadFile[] = [];
  fileListTrain: uploadFile[] = [];
  fileListPredict: uploadFile[] = [];
  selectedFiles: number[] = [];

  loadedImgUrl: string = "noImage";
  activeStep: number = 0;
  iterations: any[] = [];
  iterdatetime: string = "";
  iterdomain: string = "";
  iterid: string = "";
  _iterationId: number = -1;
  iterclasstype: string = "";
  allprojects: Project[] = [];
  formData = new FormData();
  arrayOfBlob = new Array<Blob>();
  emptyFile: File = new File(this.arrayOfBlob, "Mock.zip", { type: 'application/zip' });
  projectLoadOrCreate = "Create New";

  project: Project = {
    Name: "",
    Description: "",
    Id: "-1",
    Export: -1,
    Type: -1,
    Resource: -1,
    Thumbnail: "",
    UpdatedOn: ""
  };


  performanceCharts: PerfCharts[] = [];
  gradientBarChartConfiguration: any;
  canvas: any;
  ctx: any;
  myChart: any;
  allCharts: Chart[] = [];

  backgroundColor1: string[] = ["#00509d", "#00d2ff"];
  backgroundColor2: string[] = ["#f76775", "#f993a2"];
  backgroundColor3: string[] = ["rgb(54, 162, 235)", "#80b6f4"];

  projectName: string = "";
  processing: boolean = false;


  constructor(private restAPIService: RestApiService) { }

  ngOnInit(): void {
    this.loadAllMetaData();
    this.activeStep = 1;
  }

  @HostListener("dragover", ["$event"]) onDragOver(event: any) {
    //this.dragAreaClass = "droparea";
    event.preventDefault();
  }
  @HostListener("dragenter", ["$event"]) onDragEnter(event: any) {
    //this.dragAreaClass = "droparea";
    event.preventDefault();
  }
  @HostListener("dragend", ["$event"]) onDragEnd(event: any) {
    //this.dragAreaClass = "dragarea";
    event.preventDefault();
  }
  @HostListener("dragleave", ["$event"]) onDragLeave(event: any) {
    //this.dragAreaClass = "dragarea";
    event.preventDefault();
  }
  @HostListener("drop", ["$event"]) onDrop(event: any) {
    //this.dragAreaClass = "dragarea";
    event.preventDefault();
    event.stopPropagation();
    if (event.dataTransfer.files) {
      let files: FileList = event.dataTransfer.files;
      this.saveFiles(files);
      this.saveFilesToProjectData();
    }
  }

  loadAllProjects() {
    this.processing = true;
    this.restAPIService.loadAllProjects().subscribe(resp => {
      let results = [];
      this.allprojects = [];
      if ('success' in resp) {
        results = resp.results;
        for (let i = 0; i < results.length; i++) {
          this.allprojects.push(
            {
              Name: results[i].projectName,
              Description: results[i].Description,
              Id: results[i].id,
              Export: results[i].ExportOption,
              Resource: results[i].Resource,
              Thumbnail: results[i].thumbnail,
              Type: results[i].Type,
              UpdatedOn: results[i].UpdatedOn
            }
          )
        }
      }
      this.processing = false;
    });
  }

  loadProjectData() {
    this.processing = true;
    this.formData = new FormData();
    this.formData.append('projectId', this.project.Id);
    this.formData.append('projectStepNo', this.activeStep.toString());

    this.restAPIService.loadProjectStepsData(this.formData).subscribe(resp => {
      let results = resp.results;
      if ('success' in resp)
        if (this.activeStep == 1) {
          this.fileListTrain = [];
        }
      if (this.activeStep == 3) {
        this.fileListPredict = [];
      }

      for (let i = 0; i < results.length; i++) {
        if (this.activeStep == 1) {
          this.fileListTrain.push({ 'path': this.restAPIService.getAllSavedFiles(results[i].path), no: results[i].no, class: 'file-upload-image', file: this.emptyFile, updated: results[i].name, name: results[i].name, tag: results[i].tag })
        }
        if (this.activeStep == 3) {
          this.fileListPredict.push({ 'path': this.restAPIService.getAllSavedFiles(results[i].path), no: results[i].no, class: 'file-upload-image', file: this.emptyFile, updated: results[i].name, name: results[i].name, tag: results[i].tag })
        }
      }
      this.fileList = this.activeStep == 1 ? this.fileListTrain : this.fileListPredict;
      this.processing = false;
      console.log('fileListTrain', this.fileListTrain);
    });
  }

  createProject(data: any) {
    this.processing = true;
    console.log('data ->', data);
    data['thumbnail'] = 'type1.png';
    this.restAPIService.createnewProject(data).subscribe(resp => {
      console.log('response ->', resp);
      if ('success' in resp) {
        this.projectCreated = true;
        this.processing = false;
        this.project.Id = resp['newProjectId'];
      }
    });

  }

  onTabSelected(no: any) {
    this.activeStep = no;
    if (no == 3)
      this.loadPerformanceData();
    if (no == 1 || no == 3) {
      this.loadProjectData();
    }
  }

  loadPerformanceData() {

    this.processing = true;
    this.formData = new FormData();
    this.formData.append('projectId', this.project.Id);
    this.restAPIService.loadIterationHistory(this.formData).subscribe(resp => {
      console.log('loadIterationHistory results ->', resp);
      this.iterations = [];
      let results = resp.results;
      if ('success' in resp) {
        this.iterations = results;
        this.processing = false;
      }
    });
  }


  saveFilesToProjectData() {
    this.formData = new FormData();
    this.formData.append('projectId', this.project.Id);
    this.formData.append('resourceId', this.project.Resource.toString());
    this.formData.append('projectStep', this.activeStep.toString());
    console.log('id', this.project.Id, 'step', this.activeStep);
    this.fileList.forEach(f => {
      this.formData.append(f.name, f.file);
    });

    this.restAPIService.saveFilesToProjectData(this.formData).subscribe(resp => {
      let results = resp.results;
      if ('success' in resp) {

        if (this.activeStep == 1) {
          this.fileListTrain = [];
        }
        if (this.activeStep == 3) {
          this.fileListPredict = [];
        }

        for (let i = 0; i < results.length; i++) {
          if (this.activeStep == 1) {
            this.fileListTrain.push({ 'path': this.restAPIService.getAllSavedFiles(results[i].path), no: results[i].no, class: 'file-upload-image', file: this.emptyFile, updated: results[i].name, name: results[i].name, tag: results[i].tag })
          }
          if (this.activeStep == 3) {
            this.fileListPredict.push({ 'path': this.restAPIService.getAllSavedFiles(results[i].path), no: results[i].no, class: 'file-upload-image', file: this.emptyFile, updated: results[i].name, name: results[i].name, tag: results[i].tag })
          }
        }
      }
      this.processing = false;
    });
  }

  saveFiles(files: FileList) {

    this.processing = true;
    this.fileList = [];
    for (let i = 0; i < files.length; i++) {

      this.fileList.push({ 'no': i, 'file': files[i], 'name': files[i].name, 'path': '', 'updated': files[i].lastModified, 'class': 'file-upload-image', tag: '' });
      var reader = new FileReader();

      // Closure to capture the file information.
      reader.onload = ((theFile) => {
        return (e: any) => {
          this.fileList.filter(f => f.name == theFile.name)[0].path = e.target.result;
        };
      })(files[i]);

      // Read in the image file as a data URL.
      reader.readAsDataURL(files[i]);
    }

  }

  onFileChange(event: any) {
    console.log('event fired ->', event)
    this.saveFiles(event.target.files)
    this.saveFilesToProjectData();
  }

  removeUpload() {
    this.fileList = [];
    if (this.activeStep == 1)
      this.fileListTrain = []
    if (this.activeStep == 3)
      this.fileListPredict = [];
  }

  public currentitem: string = "";
  public isActive: boolean = false;

  selectIteration1(event: any, item: any) {
    event.preventDefault();
    console.log('iterid', item.id);
    console.log('item', item.name);
    this.currentitem = item.id;
    this.iterations.map(x => x.isSelected = false);
    this.iterations.find(x => x.id === item.id).isSelected = true;
    // this.isActive = true;
    //  console.log(HighlightRow);
    let iteration = this.iterations.find(f => f.id == item.id);
    console.log('itemidbb', iteration);
    console.log('hi');
    this.iterdatetime = iteration.UpdatedOn;
    this.iterid = iteration.name;
    this._iterationId = iteration.id;
    this.iterclasstype = "Binary Classification";

  }


  selectIteration(event: any) {
    console.log('iterid', event.target.value);
    let iteration = this.iterations.find(f => f.id == event.target.value);
    this.iterdatetime = iteration.UpdatedOn;
    this.iterid = iteration.name;
    this._iterationId = iteration.id;
    this.iterclasstype = "Binary Classification";

    let f1 = iteration.f1 * 100;
    let precision = iteration.precision * 100;
    let recall = iteration.recall * 100;

    this.performanceCharts[0].Data = [precision, 100 - precision];
    this.performanceCharts[1].Data = [recall, 100 - recall];
    this.performanceCharts[2].Data = [f1, 100 - f1];

    this.allCharts[0].data.datasets?.forEach(f => {
      f.data = this.performanceCharts[0].Data;
      console.log('c1', f.data)
    });

    this.allCharts[1].data.datasets?.forEach(f => {
      f.data = this.performanceCharts[1].Data;
      console.log('c2', f.data)
    });

    this.allCharts[2].data.datasets?.forEach(f => {
      f.data = this.performanceCharts[2].Data;
      console.log('c3', f.data)
    });

    this.allCharts[0].update();
    this.allCharts[1].update();
    this.allCharts[2].update();

  }

  loadSelectedProject(project: Project) {
    this.projectSelected = true;
    this.projectLoadOrCreate = "My Projects";
    let findProj = this.allprojects.find(p => p.Id == project.Id);
    console.log('findproj', findProj, project, this.allprojects);
    if (findProj) {
      this.project = findProj;
      this.loadProjectData();
    }
  }

  getProjectType(type: Number) {
    return this.projectTypes.find(f => f.id == type).Name;
  }

  selectTrainingImage(no: number) {
    if (this.selectedFiles.indexOf(no) != -1)
      this.selectedFiles.splice(this.selectedFiles.indexOf(no), 1);
    else
      this.selectedFiles.push(no);
    console.log('selected files ', this.selectedFiles);

  }

  addTag(data: any) {
    jQuery(".bd-example-modal-sm").modal("hide");
    jQuery('input:checkbox.chekimg').removeAttr('checked');
    console.log('tag data', data.tagname);
    this.fileListTrain.filter(f => this.selectedFiles.indexOf(f.no) > -1).forEach(f => {
      f.tag = data.tagname;
    })

    this.formData = new FormData();
    this.formData.append('projectId', this.project.Id);
    this.formData.append('projectStepNo', this.activeStep.toString());
    this.formData.append('selectedFiles', this.selectedFiles.toString());
    this.formData.append('resourceId', this.project.Resource.toString());
    this.formData.append('tagname', data.tagname);

    this.restAPIService.saveTrainTagsforProject(this.formData).subscribe(resp => {
      this.selectedFiles = [];
      console.log('train list', this.fileListTrain);

    });

  }

  deleteAllFiles() {
    this.formData = new FormData();
    this.formData.append('projectId', this.project.Id);
    this.formData.append('projectStepNo', this.activeStep.toString());
    this.formData.append('resourceId', this.project.Resource.toString());
    this.restAPIService.deleteProjectImages(this.formData).subscribe(resp => {
      this.fileListTrain = [];
      this.processing = false;
    });
  }

  deletePredFiles() {
    this.formData = new FormData();
    this.formData.append('projectId', this.project.Id);
    this.formData.append('projectStepNo', this.activeStep.toString());
    this.formData.append('resourceId', this.project.Resource.toString());
    this.restAPIService.deleteProjectImages(this.formData).subscribe(resp => {
      this.fileListPredict = [];
      this.processing = false;
    });
  }

  deleteAllIterations() {
    this.processing = true;
    this.formData = new FormData();
    this.formData.append('projectId', this.project.Id);
    this.formData.append('resourceId', this.project.Resource.toString());
    this.restAPIService.deleteProjectIterations(this.formData).subscribe(resp => {
      this.iterations = [];
      this._iterationId = -1;
      this.processing = false;
    });
  }

  deleteProject(event:any,projectId:string){
    event.preventDefault();
    this.processing = true;
    this.formData = new FormData();
    this.formData.append('projectId', projectId);
    console.log('project',projectId);
    this.restAPIService.deleteProject(this.formData).subscribe(resp => {
      this.loadAllProjects();
      // this.processing = false;
    });
  }

  startTraining() {
    this.processing = true;
    this.formData = new FormData();
    this.formData.append('projectId', this.project.Id);
    this.formData.append('projectStepNo', this.activeStep.toString());
    this.formData.append('resourceId', this.project.Resource.toString());
    this.restAPIService.runTrainingForProject(this.formData).subscribe(resp => {
      this.processing = false;
      this.nextStep.nativeElement.click();
    });
  }

  startInferencing() {
    this.processing = true;
    this.formData = new FormData();
    this.formData.append('projectId', this.project.Id);
    this.formData.append('projectStepNo', this.activeStep.toString());
    this.formData.append('resourceId', this.project.Resource.toString());
    this.formData.append('iterationId', this._iterationId.toString());
    this.restAPIService.runInferencingForProject(this.formData).subscribe(resp => {
      console.log('response prediction ->', resp)

      let results = resp.results;
      if ('success' in resp) {
        if (this.activeStep == 3) {
          this.fileListPredict = []
        }
        for (let i = 0; i < results.length; i++) {
          if (this.activeStep == 3) {
            this.fileListPredict.push({ 'path': this.restAPIService.getAllSavedFiles(results[i].FilePath), no: i, class: 'file-upload-image', file: this.emptyFile, updated: 1, name: '', tag: results[i].Tag })
          }
        }
      }
      this.processing = false;
    });

  }

  launchProjectCreate() {
    this.project = {
      Name: "",
      Description: "",
      Id: "-1",
      Export: -1,
      Type: -1,
      Resource: -1,
      Thumbnail: "",
      UpdatedOn: ""
    };

    this.projectSelected = true;
  }

  loadAllMetaData() {

    this.performanceCharts = [
      {
        Id: 'precChart',
        Name: 'Precision',
        Data: [90, 10],
        Labels: ["Precision"],
        BgColor: this.backgroundColor1
      },
      {
        Id: 'reclChart',
        Name: 'Recall',
        Data: [99, 1],
        Labels: ["Recall"],
        BgColor: this.backgroundColor2
      },
      {
        Id: 'abprecChart',
        Name: 'A.P.',
        Data: [80, 20],
        Labels: ["Absolute Precision"],
        BgColor: this.backgroundColor3
      }

    ];

    this.restAPIService.loadProjectScreenMeta().subscribe(resp => {
      let results = resp.results;
      if ('success' in resp) {
        this.loadAllProjects();
        results = resp.results;
        this.exportOptions = results.exportOptions;
        this.projectTypes = results.projectTypes;
        this.allResources = results.allResources;
      }
    });

  }

  initChartConfig(text: string) {
    this.gradientBarChartConfiguration = {
      maintainAspectRatio: false,
      legend: {
        display: true,
        position: "bottom",
        labels: {
          fontColor: "#333",
          fontSize: 9
        }
      },
      responsive: false,
      title: {
        display: true,
        position: "top",
        text: text,
        fontSize: 13,
        fontColor: "#111"
      }
    };

  }

  clearCharts() {
    this.gradientBarChartConfiguration = {};
    this.canvas = null;
    this.ctx = null;
  }
  generateCharts(perfChart: PerfCharts) {
    this.clearCharts();
    this.initChartConfig(perfChart.Name);

    this.canvas = document.getElementById(perfChart.Id);
    this.ctx = this.canvas.getContext("2d");
    var gradientStroke = this.ctx.createLinearGradient(0, 230, 0, 50);

    gradientStroke.addColorStop(1, 'rgba(37,138,128,0.4)');
    gradientStroke.addColorStop(0.4, 'rgba(37,138,128,0.2)');
    gradientStroke.addColorStop(0, 'rgba(37,138,128,0.1)'); //blue colors


    this.myChart = new Chart(this.ctx, {
      type: 'doughnut',
      data: {
        labels: perfChart.Labels,
        datasets: [{
          label: "Predictions",
          fill: true,
          backgroundColor: perfChart.BgColor,
          borderColor: 'white',
          borderWidth: 1,
          borderDash: [],
          borderDashOffset: 0.0,
          data: perfChart.Data,

        }]
      },
      options: this.gradientBarChartConfiguration
    });

    this.allCharts.push(this.myChart);
  }

}

