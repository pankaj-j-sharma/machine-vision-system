import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ProcessTrackingComponent } from './process-tracking.component';

describe('ProcessTrackingComponent', () => {
  let component: ProcessTrackingComponent;
  let fixture: ComponentFixture<ProcessTrackingComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ProcessTrackingComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ProcessTrackingComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
