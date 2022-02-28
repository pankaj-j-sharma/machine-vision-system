import { Component, HostListener, OnInit, ViewChild } from '@angular/core';
import { RestApiService } from 'src/app/services/restAPI/rest-api.service';
import { CanvasComponent } from './canvas-component';

declare var jQuery:any;

export interface uploadFile {
  no: number
  file : File 
  name:string
  path:string
  updated: number
  class: string
  tag:string
  resId : number
}

@Component({
  selector: 'app-labelling',
  templateUrl: './labelling.component.html',
  styleUrls: ['./labelling.component.scss']
})
export class LabellingComponent implements OnInit {

  fileList : uploadFile[] = [];
  fileListLabel : uploadFile[] = [];
  processing: boolean = false;
  loadedImgUrl: string = "";
  loadedImgId: number =-1;
  dragAreaClass: string = "";
  newLabel:string = "";
  allLabels:string[]=[];
  myLabels:string[]=[];
  styleColor:string="";
  labelMode:string="";
  labelColors:string[]=[];
  randomColor : string[]= [
    "rgb(49, 194, 179 ,0.2)",
    "rgb(92, 219, 211 ,0.2)",
    "rgb(9, 109, 217 ,0.2)",
    "rgb(173, 198, 255 ,0.2)",
    "rgb(146, 84, 222 ,0.2)",
    "rgb(247, 89, 171 ,0.2)",
    "rgb(255, 163, 158 ,0.2)"
  ];
  newColor: string =this.randomColor[0];
  saveDrawItems:any[]=[];
  saveDrawImages:any[]=[];
  customLabelId:number = -1;
  resourceId:number = -1;
  arrayOfBlob = new Array<Blob>();
  emptyFile:File = new File(this.arrayOfBlob, "example.zip", { type: 'application/zip' })
  allResources: any[] = [];  
  formData = new FormData();
  saveAsLabel : string = "";
  defaultSelection:number = -1;

  @ViewChild(CanvasComponent ) child!: CanvasComponent ;

  constructor(private restAPIService : RestApiService) { }

  ngOnInit(): void {
    this.dragAreaClass = "dragarea";
    this.loadMetaData();
    this.loadCustomLabelData();
    this.loadAllResources();
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
      this.saveFilesToCustomLabelData();
    }
  }  

  saveFiles(files: FileList) {
    this.processing=true;
    this.fileList = [];
    for(let i=0; i < files.length; i++){

      this.fileList.push({'no':i,'file':files[i],'name':files[i].name,'path':'','updated':files[i].lastModified , 'class':'file-upload-image',tag:'',resId:-1});        
      var reader = new FileReader();

      // Closure to capture the file information.
      reader.onload = ((theFile) => {
          return (e:any) => {
            this.fileList.filter(f=>f.name==theFile.name)[0].path=e.target.result;              
          };
      })(files[i]);

      // Read in the image file as a data URL.
      reader.readAsDataURL(files[i]);        
    }
  }

  loadCustomLabelData(){
    this.processing = true;
    this.formData = new FormData();
    let userId = this.restAPIService.getCurrentUserId();
    if (userId){
      this.formData.append('userId',userId.toString());
    }    
    this.restAPIService.loadDraftCustomLabelData(this.formData).subscribe(resp => {
      if ('success' in resp){
        let results = resp.results;
        console.log('results ',results);
        this.customLabelId = results['labelId'];
        this.resourceId = results['resourceId'];
        if (this.resourceId >-1){
          this.formData = new FormData();
          this.formData.append('resourceId',this.resourceId.toString());
          this.restAPIService.loadCustomLabelFilesData(this.formData).subscribe(resp1 => {
            if ('success' in resp){
              let results1 = resp1.results;
              for (let i = 0; i < results1.length; i++) {
                this.fileListLabel.push({'path':this.restAPIService.getAllSavedFiles(results1[i].path),no:i,class:'file-upload-image',file:this.emptyFile,updated:1,name:results1[i].name,tag:'',resId:results1[i].id});

                // load first image by default 
                if(i==0){
                  this.loadedImgUrl = this.restAPIService.getAllSavedFiles(results1[i].path);
                  this.loadedImgId = results1[i].id;        
                }

              }
            }
          });
        } 
      }
      this.processing = false;
    });
  } 

  saveFilesToCustomLabelData(){
    this.processing = true;
    this.formData = new FormData();
    this.formData.append('name','Custom labelling');
    this.formData.append('desc','Custom labelling details');
    this.formData.append('labelId',this.customLabelId.toString());
    this.formData.append('resourceId',this.resourceId.toString());
    let userId = this.restAPIService.getCurrentUserId();
    if (userId){
      this.formData.append('userId',userId.toString());
    }    
    this.fileList.forEach(f=> {
      this.formData.append(f.name,f.file);
    });

    this.restAPIService.saveFilesToCustomLabelData(this.formData).subscribe(resp => {
      console.log('saveFilesToCustomLabelData',resp);
      let results = resp.results;
      let respForm = resp.form;
      this.customLabelId = respForm.labelId;
      if('success' in resp){
        this.fileListLabel = [];
        for (let i = 0; i < results.length; i++) {
          this.fileListLabel.push({'path':this.restAPIService.getAllSavedFiles(results[i].path),no:i,class:'file-upload-image',file:this.emptyFile,updated:1,name:results[i].name,tag:'',resId:results[i].id});
        }
      }
      this.processing = false;
    });
  }

  imageSelected(event:Event,path:string,resDetailId:number){
    this.defaultSelection = -1;
    event.preventDefault();
    this.loadedImgUrl = path;
    this.loadedImgId = resDetailId;
  }

  onselectionChange(event:Event,change: any){
    event.preventDefault();
    console.log('e',change,event);
    let tmp = this.fileList.filter(f=>f.no==change);
    let reader = new FileReader();
    reader.onload = (e: any) => {
      this.loadedImgUrl=e.target.result;
    }
    if (tmp && tmp.length>0){
      let tmpFile = tmp[0].file;
      if (tmpFile!=undefined){
        reader.readAsDataURL(tmpFile);  
      }
      else{
        this.loadedImgUrl = tmp[0].path;
      }
    }  
  }

  addLabels(data:any){
    console.log('event data',data.newLabel);

    if(data.newLabel!="" && this.myLabels.length<12 && this.myLabels.indexOf(data.newLabel.toLowerCase())==-1){
      this.newLabel="";
      this.myLabels.push(data.newLabel.toLowerCase());
      for(let i=0;i < this.myLabels.length ; i++){
        if(this.myLabels[i] == this.allLabels[i]){
          continue;
        }
        else{
          this.allLabels[i] = this.myLabels[i];
          this.labelColors.push(this.getRandomColor());   
        }
      }
    }
  }

  changeBoundingBoxColor (pos:number){
    this.styleColor= this.labelColors[pos];
  }

  loadMetaData(){
    this.allLabels=["Label 1","Label 2","Label 3","Label 4","Label 5","Label 6","Label 7","Label 8","Label 9","Label 10","Label 11","Label 12"];
  }

  getRandomColor() {
    let letters = '0123456789ABCDEF';
    let color = '#';
    for (let i = 0; i < 6; i++) {
      color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
  }
 
  saveCanvasData(){
    //this.saveDrawItems =  this.child.getAllDrawnItems();
    this.child.getAllDrawnItems().forEach(item=> {
      let tmp = {id:0,x1:0,y1:0,x2:0,y2:0};
      tmp.x1 = item.x/450;
      tmp.y1 = item.y/320;
      tmp.x2 = (item.x+item.w)/450;
      tmp.y2 = (item.y+item.h) / 320;
      tmp.id = this.labelColors.indexOf(item.color);
      //tmp.name = this.myLabels[tmp.id]; 
      this.saveDrawItems.push(tmp);
    });

    this.formData = new FormData();
    this.formData.append('labelId',this.customLabelId.toString());
    this.formData.append('desc',this.saveAsLabel);
    this.formData.append('fname','custom_lbl_file');
    this.formData.append('resourceId',this.resourceId.toString());
    this.formData.append('resourceDetailId',this.loadedImgId.toString());
    this.formData.append('canvasItems',JSON.stringify(this.saveDrawItems));
    this.restAPIService.saveLabelledImages(this.formData).subscribe(resp=>{
      if('success' in resp){
        let results = resp.results;
        console.log('results ',results);
        this.loadAllResources();
      }
    });
  }

  loadAllResources(){
    this.formData = new FormData();
    this.formData.append('screen','Augmentation');
    this.restAPIService.loadAllResources(this.formData).subscribe(resp =>{
      if('success' in resp){
        let results = resp.results;
        this.allResources = results;
      }
    });
  }

  selectImageResource(event:any){
    this.processing = true;
    console.log('event',event.target.value);
    this.resourceId = event.target.value;

    this.fileListLabel = [];    
    this.formData = new FormData();
    this.formData.append('resourceId',this.resourceId.toString());
    this.restAPIService.loadCustomLabelFilesData(this.formData).subscribe(resp1 => {
      if ('success' in resp1){
        let results1 = resp1.results;
        for (let i = 0; i < results1.length; i++) {
          this.fileListLabel.push({'path':this.restAPIService.getAllSavedFiles(results1[i].path),no:i,class:'file-upload-image',file:this.emptyFile,updated:1,name:results1[i].name,tag:'',resId:results1[i].id});

          // load first image by default 
          if(i==0){
            this.loadedImgUrl = this.restAPIService.getAllSavedFiles(results1[i].path);
            this.loadedImgId = results1[i].id;  
            this.defaultSelection = i; 
          }

        }
      }
      this.processing = false;
    });

  }  

  loadSaveAsModal(){
    jQuery(".bd-example-modal-sm").modal("show");
  }

  saveAs(data:any){
    this.saveAsLabel = data.saveAsLabel;
    jQuery(".bd-example-modal-sm").modal("hide");
    this.saveCanvasData();
  }
}
