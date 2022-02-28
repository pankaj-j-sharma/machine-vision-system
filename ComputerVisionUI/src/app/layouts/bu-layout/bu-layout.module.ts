import { NgModule } from "@angular/core";
import { RouterModule } from "@angular/router";
import { BULayoutRoutes } from "./bu-layout.routing";


@NgModule({
  imports: [
    RouterModule.forChild(BULayoutRoutes),
  ],
  exports:[RouterModule],
  declarations: [],
  providers:[
  ]
})
export class BULayoutModule {}