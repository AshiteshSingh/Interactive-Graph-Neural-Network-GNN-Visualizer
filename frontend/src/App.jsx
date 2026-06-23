import React, { useState, useCallback, useEffect, useRef } from 'react';
import ReactFlow, {
  Background,
  Controls,
  applyNodeChanges,
  applyEdgeChanges,
  MarkerType,
  Handle,
  Position,
  useReactFlow,
  ReactFlowProvider
} from 'reactflow';
import 'reactflow/dist/style.css';
import axios from 'axios';
import dagre from 'dagre';
import { Layers, Zap, Activity, Cpu, Network, X, ChevronDown } from 'lucide-react';

const dagreGraph = new dagre.graphlib.Graph();
dagreGraph.setDefaultEdgeLabel(() => ({}));

const nodeWidth = 140;
const nodeHeight = 70;

const getLayoutedElements = (nodes, edges, direction = 'LR') => {
  dagreGraph.setGraph({ rankdir: direction, ranksep: 120, nodesep: 60 });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  nodes.forEach((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    node.position = {
      x: nodeWithPosition.x - nodeWidth / 2,
      y: nodeWithPosition.y - nodeHeight / 2,
    };
    node.type = 'custom';
  });

  return { nodes, edges };
};

const CustomNode = ({ data, selected }) => {
  return (
    <div className={`custom-node ${selected ? 'selected' : ''}`}>
      <Handle type="target" position={Position.Left} isConnectable={false} />
      <div className="node-icon"><Cpu size={14}/></div>
      <div className="node-content">
        <div className="node-label">{data.label}</div>
        <div className="node-flop">{data.flop.toFixed(1)} FLOPs</div>
      </div>
      <Handle type="source" position={Position.Right} isConnectable={false} />
    </div>
  );
};

const nodeTypes = { custom: CustomNode };

const mockNodes = [
  { id: '0', data: { label: 'Input', flop: 0 }, type: 'custom' },
  { id: '1', data: { label: 'Conv2D', flop: 15.2 }, type: 'custom' },
  { id: '2', data: { label: 'GeLU', flop: 0.5 }, type: 'custom' },
  { id: '3', data: { label: 'LayerNorm', flop: 0.8 }, type: 'custom' },
  { id: '4', data: { label: 'MatMul', flop: 8.4 }, type: 'custom' },
  { id: '5', data: { label: 'Softmax', flop: 1.2 }, type: 'custom' }
];

const mockEdges = [
  { id: 'e0-1', source: '0', target: '1' },
  { id: 'e1-2', source: '1', target: '2' },
  { id: 'e2-3', source: '2', target: '3' },
  { id: 'e3-4', source: '3', target: '4' },
  { id: 'e4-5', source: '4', target: '5' }
];

const mockStats = { cost: '124.5 (Base)', speedup: '1.00x', fusions: '0' };

const topologies = [
  { value: 'residual', label: 'Residual Network' },
  { value: 'chain', label: 'Sequential Chain' },
  { value: 'dense', label: 'Dense Connectivity' }
];

const CustomDropdown = ({ value, onChange }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const selectedOption = topologies.find(t => t.value === value) || topologies[0];

  return (
    <div className="custom-dropdown" ref={dropdownRef}>
      <div className="dropdown-header" onClick={() => setIsOpen(!isOpen)}>
        {selectedOption.label}
        <ChevronDown size={16} />
      </div>
      {isOpen && (
        <div className="dropdown-list">
          {topologies.map(topo => (
            <div 
              key={topo.value} 
              className={`dropdown-item ${value === topo.value ? 'active' : ''}`}
              onClick={() => {
                onChange(topo.value);
                setIsOpen(false);
              }}
            >
              {topo.label}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const FlowApp = () => {
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [topology, setTopology] = useState('residual');
  const [stats, setStats] = useState({ cost: '0.0', speedup: '1.00x', fusions: '0' });
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const { fitView } = useReactFlow();

  const fetchGraph = async (topo) => {
    setLoading(true);
    setSelectedNode(null);
    try {
      const res = await axios.get(`http://localhost:8000/api/graph?topology=${topo}`);
      const rawNodes = res.data.nodes.map(n => ({
        ...n,
        type: 'custom',
      }));
      
      const rawEdges = res.data.edges.map(e => ({
        ...e,
        type: 'smoothstep',
        animated: false,
        style: { stroke: '#4A4F62', strokeWidth: 1.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color: '#4A4F62' }
      }));

      const layouted = getLayoutedElements(rawNodes, rawEdges);
      setNodes([...layouted.nodes]);
      setEdges([...layouted.edges]);
      setStats(res.data.stats);
      
      setTimeout(() => fitView({ padding: 0.2, duration: 800 }), 50);
    } catch (err) {
      console.warn("Backend unavailable, using mock data for demo.");
      const layouted = getLayoutedElements(
        mockNodes, 
        mockEdges.map(e => ({
          ...e, type: 'smoothstep', animated: false, 
          style: { stroke: '#4A4F62', strokeWidth: 1.5 }, 
          markerEnd: { type: MarkerType.ArrowClosed, color: '#4A4F62' }
        }))
      );
      setNodes([...layouted.nodes]);
      setEdges([...layouted.edges]);
      setStats(mockStats);
      setTimeout(() => fitView({ padding: 0.2, duration: 800 }), 50);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchGraph(topology);
  }, []);

  const onNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );
  
  const onEdgesChange = useCallback(
    (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    []
  );

  const onNodeClick = (event, node) => {
    setSelectedNode(node);
  };

  const onPaneClick = () => {
    setSelectedNode(null);
  };

  const runInference = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`http://localhost:8000/api/infer`);
      const fusedIds = new Set(res.data.fused_edges);
      
      setEdges((eds) => 
        eds.map(e => {
          if (fusedIds.has(e.id)) {
            return {
              ...e,
              type: 'smoothstep',
              animated: true,
              style: { stroke: '#00F0FF', strokeWidth: 4, filter: 'drop-shadow(0 0 8px rgba(0, 240, 255, 0.8))' },
              markerEnd: { type: MarkerType.ArrowClosed, color: '#00F0FF' }
            };
          }
          return {
            ...e,
            type: 'smoothstep',
            animated: false,
            style: { stroke: '#4A4F62', strokeWidth: 1.5, opacity: 0.15 },
            markerEnd: { type: MarkerType.ArrowClosed, color: 'rgba(74, 79, 98, 0.15)' }
          };
        })
      );
      setStats(res.data.stats);
    } catch (err) {
      console.warn("Backend unavailable, using mock inference data.");
      const fusedIds = new Set(['e1-2', 'e3-4']);
      setEdges((eds) => 
        eds.map(e => {
          if (fusedIds.has(e.id)) {
            return {
              ...e, type: 'smoothstep', animated: true,
              style: { stroke: '#00F0FF', strokeWidth: 4, filter: 'drop-shadow(0 0 8px rgba(0, 240, 255, 0.8))' },
              markerEnd: { type: MarkerType.ArrowClosed, color: '#00F0FF' }
            };
          }
          return {
            ...e, type: 'smoothstep', animated: false,
            style: { stroke: '#4A4F62', strokeWidth: 1.5, opacity: 0.15 },
            markerEnd: { type: MarkerType.ArrowClosed, color: 'rgba(74, 79, 98, 0.15)' }
          };
        })
      );
      setStats({ cost: '82.3 (Optimized)', speedup: '1.51x', fusions: '2' });
    }
    setLoading(false);
  };

  return (
    <div className="app-container">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        fitView
        proOptions={{ hideAttribution: true }}
        className="dark-theme-flow"
      >
        <Background color="#1A1C23" gap={24} size={1.5} />
        <Controls className="custom-controls" showInteractive={false} />
      </ReactFlow>

      <header className="glass-panel glass-header">
        <div className="logo-section">
          <div className="logo-icon-wrapper"><Network size={24} className="logo-icon" color="#00F0FF" /></div>
          <div>
            <h1>Neural Graph Optimizer</h1>
            <p>JAX Differentiable Architecture Search</p>
          </div>
        </div>
        
        <div className="header-controls">
          <CustomDropdown 
            value={topology} 
            onChange={(val) => {
              setTopology(val);
              fetchGraph(val);
            }} 
          />
          <button className="btn-primary" onClick={() => fetchGraph(topology)} disabled={loading}>
            <Layers size={18} /> Regenerate
          </button>
          <button className="btn-accent" onClick={runInference} disabled={loading}>
            <Zap size={18} fill="#000" /> Execute JAX Optimizer
          </button>
        </div>
      </header>

      <div className="glass-panel stats-panel">
        <h3><Activity size={16}/> Inference Metrics</h3>
        <div className="stat-grid">
          <div className="stat-box">
            <span className="stat-label">Oracle Cost</span>
            <span className="stat-value">{stats.cost}</span>
          </div>
          <div className="stat-box accent">
            <span className="stat-label">Est. Speedup</span>
            <span className="stat-value">{stats.speedup}</span>
          </div>
          <div className="stat-box">
            <span className="stat-label">Fused Ops</span>
            <span className="stat-value">{stats.fusions}</span>
          </div>
        </div>
      </div>

      <div className={`glass-panel inspector-panel ${selectedNode ? 'open' : ''}`}>
        {selectedNode && (
          <>
            <button className="close-btn" onClick={() => setSelectedNode(null)}><X size={20}/></button>
            <h3><Cpu size={16}/> Node Inspector</h3>
            <div className="inspector-content">
              <div className="inspector-row">
                <span>Node ID</span>
                <strong>{selectedNode.id}</strong>
              </div>
              <div className="inspector-row">
                <span>Operation Type</span>
                <strong className="op-type">{selectedNode.data.label}</strong>
              </div>
              <div className="inspector-row">
                <span>Compute Load</span>
                <strong>{selectedNode.data.flop.toFixed(2)} FLOPs</strong>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

const App = () => (
  <ReactFlowProvider>
    <FlowApp />
  </ReactFlowProvider>
);

export default App;
