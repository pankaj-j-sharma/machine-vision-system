import {
    Component,
    Input,
    ElementRef,
    AfterViewInit,
    ViewChild,
    SimpleChanges,
    OnInit
  } from '@angular/core';
  import { fromEvent } from 'rxjs';
  import { switchMap, takeUntil, pairwise } from 'rxjs/operators';
  
  @Component({
    selector: 'app-canvas',
    template: '<canvas class="coveringCanvas" #canvas></canvas>',
    styleUrls: ['./labelling.component.scss']
  })

  export class CanvasComponent implements OnInit, AfterViewInit {

    @ViewChild('canvas') public canvas: ElementRef | undefined;
  
    @Input() public width = 500;
    @Input() public height = 500;
    @Input() public styleColor = "";
    @Input() imgSrc = "assets/img/placeholder.jpg";
    @Input() startPosition: { x: number; y: number }={x:1,y:1};
    @Input() rect:any;
    @Input() labelMode:any;

    canvasEl!: HTMLCanvasElement;

    drawnRectangle:any;
    drawItems:any[]=[];
    saveDrawItems:any[]=[];
    images:any[]=[];

    private cx: CanvasRenderingContext2D | null | undefined;
    image: any;

    
    ngOnChanges(changes: SimpleChanges) {  
        if("imgSrc" in changes && changes.imgSrc.currentValue != changes.imgSrc.previousValue ){

            if(this.drawnRectangle)
                this.drawItems.push(this.drawnRectangle);
                this.drawnRectangle=null;

            if(this.images.filter(i=>i.src==changes.imgSrc.previousValue).length == 0 )            
                this.images.push({src:changes.imgSrc.previousValue,items:this.drawItems});
            else
                this.images.filter(i=>i.src==changes.imgSrc.previousValue)[0].items=this.drawItems;

            this.drawItems=[];

            if(this.images.filter(i=>i.src==changes.imgSrc.currentValue).length>0)
                this.drawItems = this.images.filter(i=>i.src==changes.imgSrc.currentValue)[0].items;

            //this.ngAfterViewInit();
            //console.log('this.images',this.images,this.drawItems );
            this.cx = this.canvasEl.getContext('2d');
            this.cx?.clearRect(0, 0,this.width, this.height);
            this.redrawRectangles();

            
        }      

    }

    ngOnInit(): void {

    }
        
    public ngAfterViewInit() {
      this.canvasEl= this.canvas?.nativeElement;
      this.cx = this.canvasEl.getContext('2d');

      this.canvasEl.width = this.width;
      this.canvasEl.height = this.height;

      if (!this.cx) throw 'Cannot get context';
  
      this.cx.lineWidth = 1;
      this.cx.lineCap = 'round';
      this.cx.strokeStyle = '#000';
      this.rect = this.canvasEl.getBoundingClientRect();
      this.captureEvents(this.canvasEl);
      this.redrawRectangles();
    //   this.drawImageOnCanvas();

    }

    drawImageOnCanvas(){
        this.image = new Image();
        this.image.src=this.imgSrc;
        
        //scaling the image
        this.image.onload = ()=> {            
            this.cx?.drawImage(this.image, 0, 0,this.width, this.height);
        }        
    }
  
    private captureEvents(canvasEl: HTMLCanvasElement) {
      // this will capture all mousedown events from the canvas element
      fromEvent(canvasEl, 'mousedown')
        .pipe(
          switchMap((e:any) => {
              this.startPosition.x=e.clientX - this.rect.left;
              this.startPosition.y=e.clientY - this.rect.top ;
              if(this.drawnRectangle)
                  this.drawItems.push(this.drawnRectangle);

            // after a mouse down, we'll record all mouse moves
            return fromEvent(canvasEl, 'mousemove').pipe(
              // we'll stop (and unsubscribe) once the user releases the mouse
              // this will trigger a 'mouseup' event
              takeUntil(fromEvent(canvasEl, 'mouseup')),
              // we'll also stop (and unsubscribe) once the mouse leaves the canvas (mouseleave event)
              takeUntil(fromEvent(canvasEl, 'mouseleave')),
              // pairwise lets us get the previous value to draw a line from
              // the previous point to the current point
              pairwise(),
            );
          })
        )
        .subscribe((res) => {
          //const rect = canvasEl.getBoundingClientRect();
          const prevMouseEvent = res[0] as MouseEvent;
          const currMouseEvent = res[1] as MouseEvent;
  
          // previous and current position with the offset
          const prevPos = {
            x: prevMouseEvent.clientX - this.rect.left,
            y: prevMouseEvent.clientY - this.rect.top
          };
  
          const currentPos = {
            x: currMouseEvent.clientX - this.rect.left,
            y: currMouseEvent.clientY - this.rect.top
          };
  
          // this method we'll implement soon to do the actual drawing
          this.drawOnCanvas(prevPos, currentPos);
        });
    }
  
    private drawOnCanvas(
      prevPos: { x: number; y: number },
      currentPos: { x: number; y: number }
    ) {
      if (!this.cx) {
        return;
      }
        
    //   this.cx.beginPath();
  
      if (prevPos) {
        
        let frameWidth = currentPos.x - this.startPosition.x;
        let frameHeight = currentPos.y - this.startPosition.y;
        if(this.styleColor ==""){
          return;          
        }
        if(this.labelMode=="Object Detection with Bounding Boxes"){ 
            // this.cx.clearRect(this.drawnRectangle.x,this.drawnRectangle.y, this.drawnRectangle.w, this.drawnRectangle.h);
            //this.cx.clearRect(this.startPosition.x, this.startPosition.y,prevPos.x - this.startPosition.x, prevPos.y - this.startPosition.y);
            this.cx.clearRect(this.startPosition.x, this.startPosition.y,this.width, this.height);
            this.redrawRectangles();
            this.cx.beginPath();
            this.cx.rect(this.startPosition.x, this.startPosition.y, frameWidth, frameHeight);
            this.cx.strokeStyle  = this.styleColor;
            this.cx.stroke();
                // push item to save
            this.drawnRectangle = {x:this.startPosition.x,y:this.startPosition.y, w:frameWidth, h:frameHeight , color:this.styleColor};
            
        }else if(this.labelMode=="Semantic Segmentation with Masks"){
            this.cx.beginPath();
            this.cx.moveTo(prevPos.x, prevPos.y); // from
            this.cx.lineTo(currentPos.x, currentPos.y);
            this.cx.strokeStyle  = this.styleColor;
            this.cx.stroke();
            }
        
      }
    }

    redrawRectangles(){        
        this.drawItems.forEach(r=>{
          if (this.cx){
            this.cx.beginPath();
            this.cx.rect(r.x,r.y, r.w, r.h);
            this.cx.strokeStyle  = r.color;
            this.cx.stroke();            
          }
        });
    }

    getAllDrawnItems(){
      this.saveDrawItems = this.drawItems;
      this.saveDrawItems.push(this.drawnRectangle);
      return this.saveDrawItems;
    }
    
    saveDrawnItemsToFile(){
      this.drawItems.forEach(item => {
        let tmp = {x1:'',y1:'',x2:'',y2:''};
        tmp.x1 = item.x ;
        tmp.y1 = item.y;
        tmp.x2 = item.x+item.w;
        tmp.y2 = item.y+item.h;
        this.saveDrawItems.push(tmp);
      });
      console.log('items',this.saveDrawItems);
    }    
  }
  